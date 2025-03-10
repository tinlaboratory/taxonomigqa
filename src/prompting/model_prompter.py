from transformers import pipeline, AutoTokenizer
import pandas as pd
import re
import sys
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from omegaconf import OmegaConf
# from torch.nn.utils.rnn import pad_sequence
    
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

def create_prompts(df: pd.DataFrame) -> List[str]:
    return [
        f"Descrption: {row['scene_description']} Question: {row['question']} Answer:"
        for _, row in df.iterrows()
    ]

def prompt_match(inputs: List[str]) -> List[str]:
    answers = []
    for input_text in inputs:
        match = re.search(r"Answer:\s*(.*)", input_text)
        if match:
            answers.append(match.group(1))
        else:
            answers.append("")
    return answers

def batch_prompt_model(prompts: List[str], cfg) -> List[str]:
    # Initialize the model once
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.id)
    tokenizer.padding_side="left"
    text_gen_pipeline = pipeline(
        task="text-generation",
        model=cfg.model.id,
        device_map="auto",
        torch_dtype=torch.float16, 
        tokenizer=tokenizer
    )
    # else:
    #     text_gen_pipeline = pipeline(
    #         task="text-generation",
    #         model=cfg.model.id,
    #         device_map="auto",
    #         torch_dtype=torch.float16, 
    #     )
    
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
        model_outputs = text_gen_pipeline(
            batch['prompts'],
            max_new_tokens=cfg.model.max_new_tokens,
            # temperature=cfg.model.temperature,
            # top_p=cfg.model.top_p,
            do_sample=False,
            batch_size=len(batch['prompts'])
        )
        
        # Extract generated texts
        generated_texts = [output[0]['generated_text'] for output in model_outputs]
        extracted_outputs = prompt_match(generated_texts)
        
        # Place outputs in their original positions
        for orig_idx, output in zip(batch['indices'], extracted_outputs):
            outputs[orig_idx] = output
        
        print(f"Processed batch {i}/{total_batches}")
    
    return outputs

def main(cfg):
    # Load data
    df = load_csv(cfg.paths.csv_path)
    unique_image_ids = df['image_id'].unique()[:10]
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
  