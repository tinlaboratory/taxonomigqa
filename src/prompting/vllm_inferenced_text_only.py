from vllm import LLM, SamplingParams
import argparse
import pandas as pd
from omegaconf import OmegaConf
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Any
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_csv(path:str)-> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def create_prompts(df: pd.DataFrame) -> List[str]:
    return [
        f"Descrption: {row['scene_description']} Question: {row['question']} Answer:"
        for _, row in df.iterrows()
    ]

class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
        self.lengths = [len(p) for p in prompts]
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'prompt': self.prompts[idx],
            'length': self.lengths[idx],
            'idx': idx
        }
    
def load_csv(path:str)-> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    # Sort batch by length in descending order
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    return {
        'prompts': [item['prompt'] for item in batch],
        'lengths': [item['length'] for item in batch],
        'indices': [item['idx'] for item in batch]
    }

def batch_prompt_model(prompts: List[str], cfg) -> List[str]:
    llm = LLM(
    model=cfg.model.id,
    max_model_len=4096,max_num_seqs=16,enforce_eager=True,
    dtype="float16",
    device='cuda')  # Replace with your preferred model

    tok = llm.get_tokenizer()
    allowed_tokens = tok([" Yes", " No", "Yes", "No", " yes", " no", "yes", "no"], add_special_tokens=False).input_ids
    allowed_tokens = [t[0] for t in allowed_tokens]
    
    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1, allowed_token_ids=allowed_tokens, logprobs=1, stop_token_ids=None)
    
    # Initialize output array
    outputs = [""] * len(prompts)
    
    # Create dataset and dataloader
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,  # Adjust based on your CPU
        pin_memory=True  # Helps with GPU transfer
    )
    
    total_batches = len(dataloader)
    
    for i, batch in enumerate(dataloader, 1):
        model_outputs = llm.generate(batch['prompts'], sampling_params)
        # # Extract generated texts
        generated_texts = [output.outputs[0].text for output in model_outputs]
        # Place outputs in their original positions
        for orig_idx, output in zip(batch['indices'], generated_texts):
            outputs[orig_idx] = output
        print(f"Processed batch {i}/{total_batches}")
    return outputs

def main(cfg):
    # Load data
    df = load_csv(cfg.paths.csv_path)
    unique_image_ids = df['image_id'].unique()
    filtered_df = df[df['image_id'].isin(unique_image_ids)]

    filtered_df = filtered_df[filtered_df['ground_truth'].str.lower().isin(['yes', 'no'])]
    print(len(filtered_df))
    
    # Create all prompts at once
    prompts = create_prompts(filtered_df)

    # Process all prompts in batches
    outputs = batch_prompt_model(prompts, cfg)

    # Update DataFrame
    filtered_df.loc[filtered_df.index, "model_output"] = outputs
    
    # Save results
    filtered_df.to_csv(cfg.paths.output_path, index=False, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    main(cfg)
  