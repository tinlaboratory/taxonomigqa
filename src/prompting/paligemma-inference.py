from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
import os
import pandas as pd
from omegaconf import OmegaConf

def load_csv(path:str)-> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def load_gqa_data(cfg):
    df = load_csv(cfg.paths.csv_path)
    unique_image_ids = df['image_id'].unique()[:cfg.data.num_unique_images]
    filtered_df = df[df['image_id'].isin(unique_image_ids)]
    print(len(filtered_df))
    print(len(unique_image_ids))
    filtered_df = filtered_df[filtered_df['ground_truth'].str.lower().isin(['yes', 'no'])]
    
    questions = []
    images = []
    
    for i, (_,row) in enumerate(filtered_df.iterrows()):
        # if i < cfg.data.num_samples:
        image_id = row['image_id']
        question = row['question']
        img_path = os.path.join(cfg.paths.image_dir, f"{image_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        questions.append(question)
        # else:
        #     break
    return images, questions

def get_multi_modal_input(cfg):
    images, questions = load_gqa_data(cfg)
    return [{"data": img, "question": ques} for img, ques in zip(images, questions)]

def main(cfg):
    df = load_csv(cfg.paths.csv_path)
    unique_image_ids = df['image_id'].unique()[:cfg.data.num_unique_images]
    filtered_df = df[df['image_id'].isin(unique_image_ids)]
    filtered_df = filtered_df[filtered_df['ground_truth'].str.lower().isin(['yes', 'no'])]

    llm = LLM(model=cfg.model.name, 
            max_model_len=cfg.model.max_model_len, trust_remote_code=True)
    stop_token_ids = None  # Set appropriately if needed
    
    modality = "image"
    mm_inputs = get_multi_modal_input(cfg)
        
    # Prepare inputs for generation
    inputs = []
    for mm_input in mm_inputs:
        data = mm_input["data"]
        question = mm_input["question"]
        
        # input_entry = {
        #     "prompt": "answer en "+ question + "\n\n",
        #     "multi_modal_data": {
        #         modality: data
        #     },
        # }
        input_entry = {
            "prompt": "vqa2:"+ question,
            "multi_modal_data": {
                modality: data
            },
        }
        inputs.append(input_entry)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        max_tokens=cfg.sampling.max_tokens,
        stop_token_ids=stop_token_ids
    )
    
    # Perform generation
    chunk_size = 2000
    outputs = []
    for i in range(0, len(inputs), chunk_size):
        chunk = inputs[i:i + chunk_size]
        print(f"Generating chunk {i // chunk_size + 1}/{(len(inputs) + chunk_size - 1) // chunk_size}...")
        chunk_outputs = outputs = llm.generate(chunk, sampling_params=sampling_params)
        outputs.extend(chunk_outputs)
    # outputs = llm.generate(inputs, sampling_params=sampling_params)
    exit()
    for i, (index, row) in enumerate(filtered_df.iterrows()):
        if i < len(outputs):
            generated_text = outputs[i].outputs[0].text
            filtered_df.at[index, "model_output"] = generated_text
            
    filtered_df.to_csv(cfg.paths.output_path, sep='\t')

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Using vLLM for paligemma inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    main(cfg)

# python paligemma-inference.py --config config.yaml``