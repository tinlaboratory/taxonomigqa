import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def get_embedding_matrix(model_name: str, device_map: str = 'auto', embedding_type='input'):
    if 'llava' in model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=device_map
        )
    elif 'Qwen2.5-VL' in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype="auto",
            device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=device_map
        )
    model.eval()

    if embedding_type == 'input':
        emb = model.get_input_embeddings()
    elif embedding_type == 'output':
        emb = model.get_output_embeddings()
    else:
        raise ValueError("embedding_type must be 'input' or 'output'")

    # Try standard .weight, then .embedding, else error
    if hasattr(emb, 'weight'):
        emb_matrix = emb.weight.detach().to(device_map)
    elif hasattr(emb, 'embedding'):
        emb_matrix = emb.embedding.detach().to(device_map)
    else:
        raise AttributeError("Embedding object has neither 'weight' nor 'embedding' attribute.")

    print(f"Embedding matrix shape: {emb_matrix.shape}")
    return emb_matrix

def get_vocab(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_dict = tokenizer.get_vocab()
    vocab_list = [None] * (max(vocab_dict.values()) + 1)
    for word, index in vocab_dict.items():
        vocab_list[index] = word

    return vocab_dict, vocab_list

def get_vocab_dict(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()  # token string → id
    return vocab

def get_shared_token_ids(model_a, model_b):
    vocab_a = get_vocab_dict(model_a)
    vocab_b = get_vocab_dict(model_b)

    shared_tokens = set(vocab_a.keys()) & set(vocab_b.keys())
    print(f"Shared tokens between {model_a} and {model_b}: {len(shared_tokens)}")

    ids_a = [vocab_a[tok] for tok in shared_tokens]
    ids_b = [vocab_b[tok] for tok in shared_tokens]

    return ids_a, ids_b, list(shared_tokens)

def check_numerics(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaNs detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Infs detected in {name}")
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    print(f"{name} min: {min_val}, max: {max_val}")

def compare_embeddings(model_a, model_b, device='cuda', embedding_type='input', top_k=1000, save_dir='embed_results'):
    print(f"Comparing {model_a} ↔ {model_b}")
    os.makedirs(save_dir, exist_ok=True)
    
    # if file exists, skip
    if os.path.exists(os.path.join(save_dir, f"dissimilar_tokens_{sanitize_name(model_a)}_{sanitize_name(model_b)}.csv")):
        print(f"Skipping {model_a} ↔ {model_b} because it already exists")
        return None, None

    # Load embeddings
    emb_a = get_embedding_matrix(model_a, device, embedding_type)
    emb_b = get_embedding_matrix(model_b, device, embedding_type)

    # Get aligned token indices for shared token strings
    ids_a, ids_b, shared_tokens = get_shared_token_ids(model_a, model_b)

    # Convert to tensors
    emb_a_aligned = emb_a[ids_a]
    emb_b_aligned = emb_b[ids_b]

    # # Before similarity
    # check_numerics(emb_a_aligned, "emb_a_aligned")
    # check_numerics(emb_b_aligned, "emb_b_aligned")

    norm_a = emb_a_aligned.norm(dim=1)
    norm_b = emb_b_aligned.norm(dim=1)
    nonzero_mask = (norm_a > 1e-6) & (norm_b > 1e-6)

    # Only keep nonzero vectors
    emb_a_nonzero = emb_a_aligned[nonzero_mask]
    emb_b_nonzero = emb_b_aligned[nonzero_mask]

    # Cosine similarity
    cosims = F.cosine_similarity(emb_a_nonzero, emb_b_nonzero, dim=1)

    # # After similarity
    # check_numerics(cosims, "cosims")

    # print((emb_a_aligned.norm(dim=1) == 0).sum())
    # print((emb_b_aligned.norm(dim=1) == 0).sum())

    # zero_idx = (cosims == 0).nonzero(as_tuple=True)[0]
    # print("emb_a_aligned[zero_idx]:", emb_a_aligned[zero_idx])
    # print("emb_b_aligned[zero_idx]:", emb_b_aligned[zero_idx])

    # dot_products = (emb_a_aligned * emb_b_aligned).sum(dim=1)
    # print(dot_products[cosims == 0])

    avg_sim = cosims.mean().item()
    print(f"Average Cosine Similarity: {avg_sim:.4f}")

    # Plot similarity
    plt.figure(figsize=(12, 4))
    plt.plot(cosims.cpu().numpy())
    plt.title(f"Cosine Similarity of Embeddings: {model_a} vs {model_b}")
    plt.xlabel("Token Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plot_path = os.path.join(save_dir, f"plot_{sanitize_name(model_a)}_{sanitize_name(model_b)}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

    # Get top-K most dissimilar tokens
    idxs = torch.topk(-cosims, top_k).indices.cpu().numpy()
    tokens = [shared_tokens[i] for i in idxs]

    df = pd.DataFrame({
        'token_id': idxs,
        'token': tokens,
        'cosine_similarity': cosims[idxs].cpu().numpy()
    })
    csv_path = os.path.join(save_dir, f"dissimilar_tokens_{sanitize_name(model_a)}_{sanitize_name(model_b)}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Top-{top_k} most dissimilar tokens saved to {csv_path}")
    return avg_sim, df

def sanitize_name(name):
    return name.replace("/", "_").replace("-", "_")


model_pairs = [
    ("allenai/Molmo-7B-D-0924", "Qwen/Qwen2-7B"),
    ("meta-llama/Llama-3.2-11B-Vision", "meta-llama/Llama-3.1-8B"),
    ("llava-hf/llava-1.5-7b-hf", "lmsys/vicuna-7b-v1.5"),
    ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "Qwen/Qwen2-7B-Instruct"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "mistralai/Mistral-7B-Instruct-v0.2"),
    ("meta-llama/Llama-3.2-11B-Vision-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

avg_sim_results = []

for model_a, model_b in model_pairs:
    # avg_sim, _ = compare_embeddings(model_a, model_b, device=device, embedding_type='input', top_k=1000, save_dir='embed_results_filtered')
    avg_sim, _ = compare_embeddings(model_a, model_b, device=device, embedding_type='output', top_k=1000, save_dir='unembed_results_filtered')
    if avg_sim is not None:
        avg_sim_results.append(f"{model_a} vs {model_b}: {avg_sim:.6f}")
    else:
        avg_sim_results.append(f"{model_a} vs {model_b}: SKIPPED")

# Save average similarities to a text file
save_dir = 'unembed_results_filtered'  # or use the same save_dir as in compare_embeddings
os.makedirs(save_dir, exist_ok=True)
avg_sim_path = os.path.join(save_dir, "avg_cosine_similarities.txt")
with open(avg_sim_path, "w", encoding="utf-8") as f:
    for line in avg_sim_results:
        f.write(line + "\n")
print(f"Average cosine similarities saved to {avg_sim_path}")