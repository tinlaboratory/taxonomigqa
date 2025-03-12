from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
import os
import pandas as pd
from omegaconf import OmegaConf

def load_csv(path:str)-> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def load_gqa_data(cfg, start_idx, end_idx):
    df = load_csv(cfg.paths.csv_path)
    unique_image_ids = df['image_id'].unique()[:cfg.data.num_unique_images]
    filtered_df = df[df['image_id'].isin(unique_image_ids)]
    filtered_df = filtered_df[filtered_df['ground_truth'].str.lower().isin(['yes', 'no'])]
    
    questions = []
    images = []
    
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        if start_idx <= i < end_idx:
            image_id = row['image_id']
            question = row['question']
            img_path = os.path.join(cfg.paths.image_dir, f"{image_id}.jpg")
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            questions.append(question)
        elif i >= end_idx:
            break
    return images, questions, filtered_df[start_idx:end_idx]

def get_multi_modal_input(cfg, start_idx, end_idx):
    images, questions, filtered_df_chunk = load_gqa_data(cfg, start_idx, end_idx)
    return [{"data": img, "question": ques} for img, ques in zip(images, questions)], filtered_df_chunk

def main(cfg):
    df = load_csv(cfg.paths.csv_path)
    unique_image_ids = df['image_id'].unique()[:cfg.data.num_unique_images]
    filtered_df = df[df['image_id'].isin(unique_image_ids)]
    filtered_df = filtered_df[filtered_df['ground_truth'].str.lower().isin(['yes', 'no'])]

    llm = LLM(model=cfg.model.name, 
            max_model_len=cfg.model.max_model_len, trust_remote_code=True)
    stop_token_ids = None  # Set appropriately if needed
    modality = "image"
    
 
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        max_tokens=cfg.sampling.max_tokens,
        stop_token_ids=stop_token_ids,
    )

    # Perform generation in chunks
    chunk_size = 2000 # cfg.model.chunk_size
    outputs = []
    for start_idx in range(0, len(filtered_df), chunk_size):
        end_idx = start_idx + chunk_size
        mm_inputs, filtered_df_chunk = get_multi_modal_input(cfg, start_idx, end_idx)
        print(f"Length of mm_inputs: {len(mm_inputs)}")

        # Prepare inputs for generation
        inputs = []
        for mm_input in mm_inputs:
            data = mm_input["data"]
            question = mm_input["question"]
            input_entry = {
                "prompt": "vqa2:" + question, 
                "multi_modal_data": {
                    modality: data
                },
            }
            inputs.append(input_entry)
        
        print(f"Length of Inputs: {len(inputs)}")
        
        chunk_outputs = llm.generate(inputs, sampling_params=sampling_params)
        outputs.extend(chunk_outputs)

        
        for i, (index, row) in enumerate(filtered_df_chunk.iterrows()):
            if i < len(chunk_outputs):
                generated_text = chunk_outputs[i].outputs[0].text
                filtered_df.at[index, "model_output"] = generated_text
    
    filtered_df.to_csv(cfg.paths.output_path, sep='\t')

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Using vLLM for paligemma inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    main(cfg)
