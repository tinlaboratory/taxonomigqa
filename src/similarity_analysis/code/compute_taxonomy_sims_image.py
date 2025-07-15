import os
import json
import glob
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple, Dict
from minicons import scorer
import pandas as pd
import time
import argparse
from types import MethodType

import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Compute taxonomy similarities for images.")
    parser.add_argument('--nonleaf_out_pkl', type=str, default="../data/nl_node_to_embeds_po.pkl", 
                        help="Output CSV file path for similarity results.")
    parser.add_argument('--leaf_out_pkl', type=str, default="../data/leaf_node_to_embeds_po.pkl",   
                        help="Output CSV file path for similarity results.")
    parser.add_argument('--sim_csv_out', type=str, default="../data/img_similarity.tsv", 
                        help="Output CSV file path for similarity results.")
    parser.add_argument('--last_hidden_state', type=bool, default=False,
                        help="Use last_hidden_state for image representation (True) or pooler_output (False). This is incase pickle files are not found.")
    return parser.parse_args()

# ── CONFIG ──────────────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen2.5-VL-7B-Instruct" 
DEVICE           = "cuda:0"
TORCH_DTYPE      = torch.float16
BATCH_SIZE       = 32 

# Paths
TAXONOMY_PATH    = "../data/arg_hypernyms.json"
ANNOT_PATH       = "../data/combined.json"
THINGS_BASE = "/projectnb/tin-lab/yuluq/"
THINGS_PATH      = THINGS_BASE + "data/THINGS/object_images"

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_model(model_name: str):
    """Load vision–language model via minicons."""
    return scorer.VLMScorer(model_name, device=DEVICE, torch_dtype=TORCH_DTYPE)


def gather_image_paths_for_nodes(
    nodes: List[str],
    annotation_data: Dict[str, List[str]],
    things_folders: List[str],
    things_root: str
) -> Dict[str, List[str]]:
    """
    For each node, collect all THINGS image file paths.
    If `annotation_data[node]` exists, use that to map to THINGS subfolders.
    Otherwise, look for a folder matching the node name directly.
    """
    node_to_paths = {}
    for node in nodes:
        paths = []
        # first try annotated mapping:
        if node in annotation_data and annotation_data[node]:
            for cand in annotation_data[node]:
                if cand in things_folders:
                    p = glob.glob(f"{things_root}/{cand}/*")
                    paths.extend(p)
                else:
                    print(f"Warning: annotation '{cand}' for '{node}' not in THINGS folders.")
        # fallback to direct folder:
        elif node in things_folders:
            paths = glob.glob(f"{things_root}/{node}/*")
        else:
            print(f"No images found for node '{node}'")
        if paths:
            node_to_paths[node] = paths
    return node_to_paths


def log_gpu(prefix=""):
    t = torch.cuda.memory_allocated()  / 1e9
    r = torch.cuda.memory_reserved()   / 1e9
    p = torch.cuda.max_memory_allocated() / 1e9
    print(f"{prefix} | allocated {t:.2f} GB  reserved {r:.2f} GB  peak {p:.2f} GB")

def get_batch_image_representation(
    vlm,
    images: List[Image.Image],
    batch_size: int = BATCH_SIZE,
    last_hidden_state: bool = False
) -> torch.Tensor:
    """
    Returns a [N, D] tensor of mean-pooled last_hidden_state (CLS) per image.
    """
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            # preprocess
            if hasattr(vlm, "image_processor"):
                pvs = vlm.image_processor(batch, return_tensors="pt")
            else:
                pvs = vlm.tokenizer.image_processor(batch, return_tensors="pt")
            
            pixel_values = pvs["pixel_values"].to(DEVICE, TORCH_DTYPE)
            if "Qwen2.5-VL" in MODEL_NAME:
                    image_grid_thw = pvs["image_grid_thw"] 
            
            if "llava" in MODEL_NAME:
                # forward
                out = vlm.model.vision_tower(pixel_values)
                # mean-pool last_hidden_state over sequence dim → [batch, D]
                emb = out.last_hidden_state.mean(dim=1) if last_hidden_state else out.pooler_output
            elif "Qwen2.5-VL" in MODEL_NAME:
                out = vlm.model.visual(pixel_values, image_grid_thw)

                # print(f"visual out shape: {out.shape}")
                patch_counts = [int(t * h * w) for t, h, w in image_grid_thw.cpu().numpy()]
                splits = torch.split(out, patch_counts, dim=0)
                
                # mean-pool over patch dim → [batch, D]
                image_embs = torch.stack([patch_embs.mean(dim=0) for patch_embs in splits], dim=0)
                emb = image_embs

            all_embs.append(emb.cpu())
            
            # After processing each batch (for Qwen2.5-VL due to OOM issues)
            del batch, pvs, pixel_values, out, splits, image_embs  # or any other large tensors
            torch.cuda.empty_cache()
            gc.collect()

    if not all_embs:
        return torch.empty((0,0))
    return torch.cat(all_embs, dim=0)

def get_image_embs_for_nodes(node_to_paths, model, last_hidden_state: bool = False)-> dict:
    """
    Take node-to-paths dictionary as input. Get the image embs for all nodes in the input list, return a dictionary of node-to-embs
    """
    node_to_embs = {}
    for node, paths in tqdm(node_to_paths.items(), desc="Loading images"):
       imgs = [Image.open(p).convert("RGB").resize((448, 448)) for p in paths] # resize to 448x448 (Qwen2.5-VL)
       embs = get_batch_image_representation(model, imgs, last_hidden_state=last_hidden_state)
       print(f"Node: {node}, Number of images: {len(imgs)}, Embedding shape: {embs.shape}")
       node_to_embs[node] = embs
    return node_to_embs

# ── MAIN ───────────────────────────────────────────────────────────────────────
def get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor

def compute_pairwise_similarity(leaf_embeds, non_leaf_embeds, take_mean=False):
        """
        Compute pairwise cosine similarity between leaf and non-leaf embeddings.
        leaf_embs: [N_leaf_images, D]
        non_leaf_embs: [N_nonleaf_images, D]
        """
        # normalized features
        if take_mean:
            non_leaf_embeds = non_leaf_embeds.mean(dim=0, keepdim=True)
        leaf_embeds = leaf_embeds / get_vector_norm(leaf_embeds)
        non_leaf_embeds = non_leaf_embeds / get_vector_norm(non_leaf_embeds)
        # compute pairwise similarity
        sim = torch.matmul(leaf_embeds, non_leaf_embeds.T) # [N_leaf, N_nonleaf]

        sim = torch.mean(sim, dim=1) # [N_nonleaf]
        sim = torch.mean(sim, dim=0) # [1]
        return sim

def forward_with_pre_merger(self, hidden_states, grid_thw):
    # Same as your current code until before the merger
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    # Pass through merger.norm (RMSNorm) but skip MLP
    normalized_hidden_states = self.merger.ln_q(hidden_states)

    return normalized_hidden_states

def get_concept_pairs(hyper_data):
    return (
        (orig_arg, ancestor)
        for orig_arg, ancestors in hyper_data.items()
        for ancestor in ancestors
    )
    pass
def main(args):
    # 1. Load taxonomy 
    # check if pkl file already exists, load them if they do
    with open(TAXONOMY_PATH, "r") as f:
        hyper_data = json.load(f)    # leaf → [ancestors...]

    # 2. Identify non-leaf (all ancestors) and leaf (keys of hyper_data)
    non_leaf_nodes = set(
        anc
        for ancestors in hyper_data.values()
        for anc in ancestors
    )
    leaf_nodes = list(hyper_data.keys())

    # 3. Load or create image repsentations
    if os.path.exists(args.nonleaf_out_pkl) and os.path.exists(args.leaf_out_pkl):
        print("nl_node_to_embeds.pkl and leaf_node_to_embeds.pkl already exist. Loading")
        with open(args.nonleaf_out_pkl, "rb") as f:
            nl_node_to_embeds = pickle.load(f)
        with open(args.leaf_out_pkl, "rb") as f:
            leaf_node_to_embeds = pickle.load(f)
    else:
        # load annotations
        with open(ANNOT_PATH, "r") as f:
            annot_data = json.load(f)
        things_folders = os.listdir(THINGS_PATH)

        # Gather image paths
        print("Gathering non-leaf image paths…")
        nl_paths = gather_image_paths_for_nodes(
            list(non_leaf_nodes), annot_data, things_folders, THINGS_PATH
        ) # dictionary of non-leaf node to image paths
        print(f"Found images for {len(nl_paths)}/{len(non_leaf_nodes)} non-leaf nodes.")

        print("Gathering leaf image paths…")
        leaf_paths = gather_image_paths_for_nodes(
            leaf_nodes, annot_data, things_folders, THINGS_PATH
        ) # dictionary of leaf node to image paths
        print(f"Found images for {len(leaf_paths)}/{len(leaf_nodes)} leaf nodes.")

        # Load model
        print("Loading model…")
        vlm = load_model(MODEL_NAME)

        if "Qwen2.5-VL" in MODEL_NAME:
            # Patch the method onto your loaded model's vision tower
            vlm.model.visual.forward = MethodType(forward_with_pre_merger, vlm.model.visual)
            if hasattr(vlm.model, "model"): # Handling OOM issues for Qwen2.5-VL
                del vlm.model.model  # This is often the language model part
                torch.cuda.empty_cache()
                gc.collect()

        vlm.model.to(DEVICE)
        vlm.model.eval()

        # Compute and get image representations for each node 
        print("Computing image representations for non-leaf nodes…")
        # output is a stacked tensor of shape [N, D] where N is the number of images and D is the embedding dimension
        nl_node_to_embeds = get_image_embs_for_nodes(nl_paths, vlm, last_hidden_state=args.last_hidden_state)
        print("Computing image representations for leaf nodes…")
        leaf_node_to_embeds = get_image_embs_for_nodes(leaf_paths, vlm, last_hidden_state=args.last_hidden_state)
        # store the nl_node_to_embeds and leaf_node_to_embeds in a pickle file
        with open(args.nonleaf_out_pkl, "wb") as f:
            pickle.dump(nl_node_to_embeds, f)
        with open(args.leaf_out_pkl, "wb") as f:
            pickle.dump(leaf_node_to_embeds, f)
        print("Image representations computed and saved.")
    loop_start_time = time.time()

    # 4. Compute cosine similarity 
    new_rows = []
    # for index, row in tqdm(llava_acc_df.iterrows(), total=len(llava_acc_df), desc="Calculating Similarities"):
    for pair in get_concept_pairs(hyper_data):
        # concept1 = str(row['concept1'])
        # concept2 = str(row['concept2'])
        concept1, concept2 = pair
        print(f"\nProcessing pair: {concept1} and {concept2}")
        if concept1 not in leaf_nodes:
            print(f"Skipping pair ({concept1}, {concept2}): concept1 is not a leaf node.")
            continue
        if concept1 not in leaf_node_to_embeds or concept2 not in nl_node_to_embeds:
            print(f"Skipping pair ({concept1}, {concept2}): one or both concepts have no images.")
            continue
        img_emb_concept1 = leaf_node_to_embeds[concept1]
        img_emb_concept2 = nl_node_to_embeds[concept2]
        sim_score_mean = compute_pairwise_similarity(img_emb_concept1, img_emb_concept2, take_mean=True)
        sim_score = compute_pairwise_similarity(img_emb_concept1, img_emb_concept2, take_mean=False)
        # new_row = row.to_dict()
        new_row = {
            'concept1': concept1,
            'concept2': concept2,
            'similarity_Mean': sim_score_mean.item(),
            'similarity_pairwise': sim_score.item(),
            'category': hyper_data[concept1][-1]
        }
        new_rows.append(new_row)
    loop_end_time = time.time()
    updated_df = pd.DataFrame(new_rows)

    # Save the updated DataFrame to a new CSV file
    updated_df.to_csv(args.sim_csv_out, sep='\t', index=False)
    print(f"Done! Similarities saved to {args.sim_csv_out}.")
    print(f"Time taken for loop: {loop_end_time - loop_start_time} seconds")

if __name__ == "__main__":    
    args = parse_args()
    main(args)

