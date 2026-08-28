import argparse
import random
import clip.model
import numpy as np
import pickle as pkl
import json
import time
import subprocess
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Callable, Union, Dict

import clip
from clip.model import CLIP, ModifiedResNet, VisionTransformer

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from pcbm.learn_concepts_multimodal import *
from pcbm.data import get_dataset
from pcbm.concepts import ConceptBank
from pcbm.models import PosthocLinearCBM, get_model

from captum.attr import visualization, GradientAttribution, LayerAttribution
from utils import *



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universal-seed", default=int(time.time()), type=int, help="Universal random seed")
    
    parser.add_argument("--backbone-ckpt", required=True, type=str, help="Path to the backbone ckpt")
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")

    parser.add_argument("--pcbm-arch", default="pcbm", type=str)
    parser.add_argument("--pcbm-ckpt", required=True, type=str, help="Path to the PCBM checkpoint")
    parser.add_argument("--explain-method", required=True, type=str)
    parser.add_argument("--concept-pooling", default="max_pooling_class_wise", type=str)
    parser.add_argument("--concept-target", default="", type=str)
    parser.add_argument("--class-target", default="", type=str)
    parser.add_argument("--top-k", default=0, type=int,
                        help="Per-sample mode: generate heatmaps for each sample's top-k concepts. "
                             "When >0, --concept-target is ignored.")
    
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    parser.add_argument("--exp-name", default=str(datetime.now().strftime("%Y%m%d%H%M%S")), type=str)
    parser.add_argument('--save-100-local', action='store_true')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--sample-ids', default=None, type=str,
                        help="Path to txt file with image IDs (one per line). Only these images are processed.")

    return parser.parse_args()


class concept_select_func:
    @staticmethod
    def cifar10(model_context: model_pipeline,
                concept_target:str):
        targeted_concept_idx = model_context.concept_bank.concept_names.index(concept_target)
        return targeted_concept_idx
    
    @staticmethod
    def cub(model_context: model_pipeline,
                concept_target:str):
        if hasattr(CUB_features, concept_target):
            # trick to get the device of a nn.Module
            return torch.arange(getattr(CUB_features, concept_target)[0], getattr(CUB_features, concept_target)[1] + 1)\
                .to(next(model_context.posthoc_layer.parameters()).device)
        
        return model_context.concept_bank.concept_names.index(int(concept_target))
    
    @staticmethod
    def rival10(model_context: model_pipeline,
                concept_target:str):
        targeted_concept_idx = model_context.concept_bank.concept_names.index(concept_target)
        return targeted_concept_idx

    @staticmethod
    def awa2(model_context: model_pipeline,
             concept_target: str):
        # AwA2 concepts are single named attributes (e.g. "black", "furry")
        return model_context.concept_bank.concept_names.index(concept_target)
    
    
def main(args:argparse.Namespace):
    set_random_seed(args.universal_seed)
    concept_bank, backbone, dataset, model_context, model = load_model_pipeline(args)
    model.eval()

    explain_algorithm:GradientAttribution = getattr(model_explain_algorithm_factory,
                                                    args.explain_method)(forward_func=model.encode_as_concepts,
                                                                        model = model)
    explain_algorithm_forward:Callable = getattr(model_explain_algorithm_forward, args.explain_method)
    attribution_pooling:Callable[..., torch.Tensor] = getattr(attribution_pooling_forward, args.concept_pooling)

    if args.top_k == 0:
        targeted_concept_idx = getattr(concept_select_func, args.dataset)(model_context, args.concept_target)
        args.logger.info(targeted_concept_idx)

    # build concept name list for metadata
    if hasattr(concept_bank, 'concept_names'):
        concept_names_list = concept_bank.concept_names
    else:
        concept_names_list = [str(k) for k in concept_bank.keys()]

    # load sample ID filter if provided
    sample_ids = None
    if args.sample_ids is not None:
        with open(args.sample_ids) as f:
            sample_ids = set(line.strip() for line in f if line.strip())
        args.logger.info(f"Filtering to {len(sample_ids)} sample IDs from {args.sample_ids}")

    def _get_img_id(loader, idx):
        ds = loader.dataset
        if hasattr(ds, 'samples'):
            img_path = ds.samples[idx][0]
        elif hasattr(ds, 'data') and isinstance(ds.data[idx], dict):
            img_path = ds.data[idx]['img_path']
        else:
            return str(idx)
        return os.path.splitext(os.path.basename(img_path))[0]

    def _heatmap_for_concept(batch_X_req, concept_int_idx):
        """Compute and return attribution for one concept index (int)."""
        attr = explain_algorithm_forward(batch_X=batch_X_req,
                                         explain_algorithm=explain_algorithm,
                                         target=concept_int_idx)
        attr = attribution_pooling(batch_X=batch_X_req,
                                   attributions=attr,
                                   concept_idx=concept_int_idx,
                                   pcbm_net=model)
        return attr

    # when sample_ids are provided, search test + train loaders so val-split images are found too
    loaders = [dataset.test_loader]
    if sample_ids is not None and hasattr(dataset, 'train_loader') and dataset.train_loader is not None:
        loaders.append(dataset.train_loader)

    count = 0
    done = False
    for loader in loaders:
        if done:
            break
        for idx, data in tqdm(enumerate(loader), total=len(loader)):
            # AwA2 returns (img, label, attrs); CUB returns (img, label)
            batch_X = data[0].to(args.device)
            batch_Y = data[1].to(args.device)

            img_id = _get_img_id(loader, idx)

            # filter by sample ID if provided
            if sample_ids is not None and img_id not in sample_ids:
                continue

            if args.class_target != "" and dataset.idx_to_class[batch_Y.item()] != args.class_target:
                continue

            # ── PER-SAMPLE TOP-K MODE ─────────────────────────────────────────
            if args.top_k > 0:
                if count >= 100:
                    done = True
                    break

                # forward pass: get concept scores + predicted class
                with torch.no_grad():
                    fwd = model(batch_X)
                    class_logits, concept_scores = fwd[0], fwd[1]
                predicted_class_idx = int(class_logits.argmax(dim=1).item())
                top_k_indices = concept_scores.squeeze(0).argsort(descending=True)[:args.top_k].cpu().tolist()
                top_concept_names = [str(concept_names_list[i]) for i in top_k_indices]

                save_to = os.path.join(args.save_path, f"{args.explain_method}/{img_id}")
                os.makedirs(save_to, exist_ok=True)

                # save original image once
                from utils.visual_utils import reduce_tensor_as_numpy
                from PIL import Image as PILImage
                orig_np = reduce_tensor_as_numpy(batch_X.detach())
                PILImage.fromarray((orig_np * 255).astype(np.uint8)).save(
                    os.path.join(save_to, f"{img_id}-original.jpg"))

                heatmaps = {}
                for rank, c_idx in enumerate(top_k_indices):
                    batch_X_g = batch_X.detach().requires_grad_(True)
                    attr = _heatmap_for_concept(batch_X_g, c_idx)
                    heatmaps[c_idx] = attr.detach().cpu()
                    c_name = str(concept_names_list[c_idx])
                    # viz_attn also saves original; we rename/skip the duplicate
                    viz_attn(batch_X_g.detach(),
                             attr,
                             blur=True,
                             prefix=f"rank{rank:02d}_{c_name}",
                             save_to=save_to)
                    # remove the duplicate original saved by viz_attn
                    dup = os.path.join(save_to, f"rank{rank:02d}_{c_name}-original_image.jpg")
                    if os.path.exists(dup):
                        os.remove(dup)
                    try:
                        captum_vis_attn(batch_X_g.detach(),
                                        attr,
                                        title=f"{dataset.idx_to_class[batch_Y.item()]} | rank{rank} {c_name}",
                                        save_to=os.path.join(save_to, f"rank{rank:02d}_{c_name}-captum.jpg"))
                    except:
                        pass

                torch.save({
                    "fname": img_id,
                    "predicted_class": predicted_class_idx,
                    "predicted_class_name": dataset.idx_to_class[predicted_class_idx],
                    "gt_class_name": dataset.idx_to_class[batch_Y.item()],
                    "concept_scores": concept_scores.detach().cpu(),
                    "top_concepts": top_k_indices,
                    "top_concept_names": top_concept_names,
                    "heatmaps": heatmaps,
                }, os.path.join(save_to, f"{img_id}.pt"))

                count += 1
                continue

            # ── PER-CONCEPT MODE (original) ───────────────────────────────────
            batch_X.requires_grad_(True)

            attributions:torch.Tensor = explain_algorithm_forward(batch_X=batch_X,
                                                                  explain_algorithm=explain_algorithm,
                                                                  target=targeted_concept_idx)
            attributions = attribution_pooling(batch_X = batch_X,
                                               attributions = attributions,
                                               concept_idx = targeted_concept_idx,
                                               pcbm_net = model)

            # forward pass for concept scores and predicted class (no grad needed)
            with torch.no_grad():
                fwd = model(batch_X.detach())
                class_logits, concept_scores = fwd[0], fwd[1]
            predicted_class_idx = int(class_logits.argmax(dim=1).item())
            top_concepts = concept_scores.squeeze(0).argsort(descending=True).cpu()

            if args.save_100_local:
                if count >= 100:
                    done = True
                    break
                save_to = os.path.join(args.save_path, f"{args.explain_method}/{args.concept_target}_images")
                os.makedirs(save_to, exist_ok=True)
                viz_attn(batch_X,
                        attributions,
                        blur=True,
                        prefix=img_id,
                        save_to=save_to)
                try:
                    captum_vis_attn(batch_X,
                                    attributions,
                                    title=f"{dataset.idx_to_class[batch_Y.item()]}-attributions: {args.concept_target}",
                                    save_to=os.path.join(save_to, f"{img_id}-captum-image.jpg"))
                except:
                    pass

                # save structured .pt dict
                concept_idx_val = targeted_concept_idx.tolist() if isinstance(targeted_concept_idx, torch.Tensor) else targeted_concept_idx
                torch.save({
                    "fname": img_id,
                    "attribution": attributions.detach().cpu(),
                    "concept_scores": concept_scores.detach().cpu(),
                    "top_concepts": top_concepts,
                    "predicted_class": predicted_class_idx,
                    "predicted_class_name": dataset.idx_to_class[batch_Y.item()],
                    "gt_class_name": dataset.idx_to_class[batch_Y.item()],
                    "concept_name": args.concept_target,
                    "concept_idx": concept_idx_val,
                }, os.path.join(save_to, f"{img_id}.pt"))

                count += 1

            else:
                for i in range(batch_Y.size(0)):
                    print(f"ground truth: {dataset.idx_to_class[batch_Y[i].item()]}")
                topK_concept_to_name(args, model, batch_X)
                viz_attn(batch_X,
                        attributions,
                        blur=True,
                        save_to=None)
                captum_vis_attn(batch_X,
                            attributions,
                            title=f"{dataset.idx_to_class[batch_Y.item()]}-attributions: {args.concept_target}",
                            save_to=None)

    # save concept index → name mapping once per run
    concept_index_path = os.path.join(args.save_path, "concept_names.json")
    with open(concept_index_path, "w") as f:
        json.dump({str(i): str(n) for i, n in enumerate(concept_names_list)}, f, indent=2)
    args.logger.info(f"Concept names saved to {concept_index_path}")

    # original_Xs = torch.concat(original_Xs, dim = 0)
    # batch_Ys = torch.concat(batch_Ys, dim = 0)
    # adversarial_Xs = torch.concat(adversarial_Xs, dim = 0)
    
    # evaluate_adversarial_sample(ori_adv_pair(
    #     original_X=(adversarial_Xs - original_Xs),
    #     # adversarial_X=adversarial_Xs,
    # ))
    
if __name__ == "__main__":
    args = config()
    args.save_path = os.path.join("./outputs/evals", args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)
    
    args_dict = vars(args)
    args_json = json.dumps(args_dict, indent=4)
    
    args.logger = common_utils.create_logger(log_file = os.path.join(args.save_path, "exp_log.log"))
    args.logger.info(args_json)
    args.logger.info(f"universal seed: {args.universal_seed}")
    if not torch.cuda.is_available():
        args.device = "cpu"
        args.logger.info(f"GPU devices failed. Change to {args.device}")
    main(args)

    if args.zip:
        command = ["zip", "-r", args.save_path + ".zip", args.save_path]
        subprocess.run(command, check=True)
    