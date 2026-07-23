# -*- coding: utf-8 -*-
# Concept bank builder for AwA2 using CREAM backbone.
# Builds one CAV per attribute (85 total) using positive/negative image splits.
# Each concept = one row in adjacency matrix = one attribute (e.g. "black", "furry", "ocean")
#
# USAGE:
#   python training_tools/learn_concepts_awa2.py \
#     --backbone-name resnet18_cream \
#     --backbone-ckpt ~/GUIDE-cbm-WIP/models/resnet18_finetuned.pth \
#     --out-dir concept_banks/ \
#     --C 0.1 --n-samples 50

import os
import pickle
import torch
import argparse
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import load_backbone
from pcbm.concepts import learn_concept_bank
from utils.constants import dataset_constants


# ========================================================================
class AwA2ConceptDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of image paths
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img  # pcbm get_embeddings expects plain tensor, not (img, label)


# ========================================================================
def get_awa2_concept_loaders(pkl_path, predicates_path, preprocess,
                              n_samples=50, batch_size=100, num_workers=4, seed=24):
    np.random.seed(seed)

    with open(pkl_path, 'rb') as f:
        all_features = pickle.load(f)

    with open(predicates_path, 'r') as f:
        predicates = [line.strip() for line in f.readlines()]

    # Group images by concept presence (attribute_label[i] == 1)
    concept_loaders = {}
    for conc_idx, conc_name in enumerate(predicates):
        pos_paths = [d['img_path'] for d in all_features if d['attribute_label'][conc_idx] == 1]
        neg_paths = [d['img_path'] for d in all_features if d['attribute_label'][conc_idx] == 0]

        # pcbm splits loader output into train (first n_samples) and val (next n_samples)
        # so each loader must provide 2*n_samples images
        if len(pos_paths) < 2 * n_samples or len(neg_paths) < 2 * n_samples:
            print(f"  [SKIP] {conc_name}: not enough samples "
                  f"(pos={len(pos_paths)}, neg={len(neg_paths)}, need {2*n_samples} each)")
            continue

        np.random.shuffle(pos_paths)
        np.random.shuffle(neg_paths)

        pos_loader = DataLoader(
            AwA2ConceptDataset(pos_paths[:2 * n_samples], preprocess),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)
        neg_loader = DataLoader(
            AwA2ConceptDataset(neg_paths[:2 * n_samples], preprocess),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        concept_loaders[conc_name] = {'pos': pos_loader, 'neg': neg_loader}

    return concept_loaders


# ========================================================================
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-ckpt", required=True, type=str)
    parser.add_argument("--backbone-name", default="resnet18_cream", type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=24, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--C", nargs="+", default=[0.1], type=float)
    parser.add_argument("--n-samples", default=50, type=int)
    parser.add_argument("--pkl-path", default=None, type=str,
                        help="Path to train pkl (default: AWA2_PROCESSED_DIR/train_seed_44.pkl)")
    parser.add_argument("--predicates-path", default=None, type=str,
                        help="Path to predicates.txt (default: AWA2_PROCESSED_DIR/predicates.txt)")
    return parser.parse_args()


# ========================================================================
def main():
    args = config()
    if args.pkl_path is None:
        args.pkl_path = os.path.join(dataset_constants.AWA2_PROCESSED_DIR, 'train_seed_44.pkl')
    if args.predicates_path is None:
        args.predicates_path = os.path.join(dataset_constants.AWA2_PROCESSED_DIR, 'predicates.txt')

    backbone_res = load_backbone(args)
    backbone  = backbone_res.backbone_model
    preprocess = backbone_res.preprocess

    print(f"Building AwA2 concept bank with {args.backbone_name}...")
    concept_loaders = get_awa2_concept_loaders(
        args.pkl_path, args.predicates_path, preprocess,
        n_samples=args.n_samples, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed)

    concept_libs = {C: {} for C in args.C}
    skipped = []
    for conc_name, loaders in concept_loaders.items():
        pos_size = len(loaders['pos'].dataset)
        neg_size = len(loaders['neg'].dataset)
        print(f"  [{conc_name}] pos={pos_size} neg={neg_size}", flush=True)
        try:
            cav_info = learn_concept_bank(loaders['pos'], loaders['neg'], backbone,
                                          args.n_samples, args.C, device=args.device)
            for C in args.C:
                concept_libs[C][conc_name] = cav_info[C]
                train_acc = cav_info[C][1]
                val_acc   = cav_info[C][2]
            print(f"    -> train_acc={train_acc:.3f}  val_acc={val_acc:.3f}", flush=True)
        except Exception as e:
            print(f"    [ERROR] {conc_name}: {e}", flush=True)
            skipped.append(conc_name)

    print(f"\nDone. Built {sum(len(v) for v in concept_libs.values())} CAVs, "
          f"skipped {len(skipped)}: {skipped}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    for C in concept_libs:
        lib_path = os.path.join(args.out_dir,
                                f"awa2_{args.backbone_name}_{C}_{2*args.n_samples}.pkl")
        with open(lib_path, 'wb') as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved: {lib_path} ({len(concept_libs[C])} concepts)")


if __name__ == "__main__":
    main()
