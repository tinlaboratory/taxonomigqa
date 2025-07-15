import sys
sys.path.append(".")

import argparse
import csv
import torch
import random
import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_rel
from transformers import AutoTokenizer, AutoProcessor
from src.similarity_analysis.code.embeddings_analysis import get_embedding_matrix


model_pairs = [
    ("allenai/Molmo-7B-D-0924", "Qwen/Qwen2-7B"),
    ("meta-llama/Llama-3.2-11B-Vision", "meta-llama/Llama-3.1-8B"),
    ("llava-hf/llava-1.5-7b-hf", "lmsys/vicuna-7b-v1.5"),
    ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "Qwen/Qwen2-7B-Instruct"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "mistralai/Mistral-7B-Instruct-v0.2"),
    ("meta-llama/Llama-3.2-11B-Vision-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
]


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    row_dicts = []
    stats_row_dicts = []

    for model_pair in model_pairs:
        model_diff_lists = []
        for model in model_pair:
            if args.emb_unemb == 'emb':
                emb = get_embedding_matrix(model, device, 'input')
            elif args.emb_unemb == 'unemb':
                emb = get_embedding_matrix(model, device, 'output')
            if "molmo" in model.lower():
                processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
                tokenizer = processor.tokenizer
            else:
                # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

            # print(emb.shape)
            # print(len(tokenizer))

            skipped = 0
            df = pd.read_csv('src/embedding_analysis/data/unique_pos_neg.csv')
            target_hypernym_sims = []
            target_neg_avg_sims = []
            target_random_sims = []
            sim_diffs = []
            for _, row in df.iterrows():
                target = row['orig_target']
                hypernym = row['hypernym']
                negs = row['neg1'], row['neg2'], row['neg3'], row['neg4']

                target_idx = tokenizer(f"the {target}", add_special_tokens=False).input_ids[1]
                hypernym_idx = tokenizer(f"the {hypernym}", add_special_tokens=False).input_ids[1]
                negs_idx = [tokenizer(f"the {neg}", add_special_tokens=False).input_ids[1] for neg in negs]
                random_idx = random.sample(range(0, len(tokenizer)), 1)[0]
                
                target_emb = emb[target_idx]
                hypernym_emb = emb[hypernym_idx]
                neg_embs = [emb[neg_idx] for neg_idx in negs_idx]
                random_emb = emb[random_idx]

                target_hypernym_sims.append(torch.cosine_similarity(target_emb, hypernym_emb, dim=0).item())
                neg_sims = [torch.cosine_similarity(target_emb, neg_emb, dim=0).item() for neg_emb in neg_embs]
                target_neg_avg_sims.append(sum(neg_sims) / len(neg_sims))
                target_random_sims.append(torch.cosine_similarity(target_emb, random_emb, dim=0).item())
                sim_diffs.append(target_hypernym_sims[-1] - target_neg_avg_sims[-1])

            
            print(f"====Model: {model}====")
            assert len(target_hypernym_sims) == len(target_neg_avg_sims), "Length mismatch between target-hypernym and target-neg samples."
            print(f"Average cosine similarity between target and hypernym: {sum(target_hypernym_sims) / len(target_hypernym_sims):.4f}")
            print(f"Average cosine similarity between target and negative samples: {sum(target_neg_avg_sims) / len(target_neg_avg_sims):.4f}")
            print(f"Average cosine similarity between target and random samples: {sum(target_random_sims) / len(target_random_sims):.4f}")
            model_diff_lists.append(sim_diffs)
            print(f"diff(sim(target,hypernym), sim(target,avg(neg))): {np.mean(sim_diffs)}")
            row_dict = {
                'model': model,
                'avg_hypernym_sim': sum(target_hypernym_sims) / len(target_hypernym_sims),
                'avg_neg_sim': sum(target_neg_avg_sims) / len(target_neg_avg_sims),
                'avg_random_sim': sum(target_random_sims) / len(target_random_sims),
                'avg_diff': np.mean(sim_diffs)
            }
            row_dicts.append(row_dict)

        print(f"====Model Pair: {model_pair}====")
        ttest_result = ttest_rel(model_diff_lists[0], model_diff_lists[1], alternative='two-sided')
        print()
        stats_row_dict = {
            'model_vlm': model_pair[0],
            'model_lm': model_pair[1],
            'vlm_sim_avg_diff': row_dicts[-2]['avg_diff'],
            'lm_sim_avg_diff': row_dicts[-1]['avg_diff'],
            't': ttest_result.statistic,
            'p': ttest_result.pvalue
        }
        stats_row_dicts.append(stats_row_dict)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f'{args.emb_unemb}_analysis.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=row_dicts[-1].keys())
        writer.writeheader()
        for row_dict in row_dicts:
            writer.writerow(row_dict)
    
    with open(os.path.join(args.results_dir, f'{args.emb_unemb}_analysis_stats.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=stats_row_dicts[-1].keys())
        writer.writeheader()
        for stats_row_dict in stats_row_dicts:
            writer.writerow(stats_row_dict)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--emb_unemb', type=str, default='emb', help='Embedding or unembedding')
    args.add_argument('--results_dir', type=str, default='data/results/embedding-analysis/', help='Directory to save results')
    args = args.parse_args()
    main(args)
