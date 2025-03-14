from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
import os
import pandas as pd
from omegaconf import OmegaConf

# LLama 3.2
def run_mllama(question: str, modality: str):
    assert modality == "image"

    model_name = "meta-llama/Llama-3.2-11B-Vision"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    
    prompt = f"<|begin_of_text|>{question}"
    stop_token_ids = None
    return prompt, stop_token_ids

# Molmo
def run_molmo_D(question, modality):
    assert modality == "image"

    model_name = "allenai/Molmo-7B-D-0924"

    prompt = question
    stop_token_ids = None
    return prompt, stop_token_ids

def run_molmo_O(question, modality):
    assert modality == "image"

    model_name = "allenai/Molmo-7B-O-0924"

    prompt = question
    stop_token_ids = None
    return prompt, stop_token_ids

# LLaVA-1.5
def run_llava(question: str, modality: str):
    assert modality == "image"

    prompt = f"USER: \n{question}\nASSISTANT:"
    stop_token_ids = None
    return prompt, stop_token_ids

# BLIP-2
def run_blip2(question: str, modality: str):
    # assert modality == "image"

    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = f"Question: {question} Answer:"
    stop_token_ids = None
    return prompt, stop_token_ids

model_example_map = {
    "blip2": run_blip2,
    "llava": run_llava,
    "mllama": run_mllama,
    "molmo_D": run_molmo_D,
    "molmo_O": run_molmo_O,
}

def get_llm(cfg, model_name)->LLM:
    if model_name == "mllama":
        llm = LLM(
            model="meta-llama/Llama-3.2-11B-Vision",
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
            dtype="float16",
        )
        return llm
    elif model_name == "molmo_O":
        llm = LLM(
            model="allenai/Molmo-7B-O-0924",
            trust_remote_code=True,
            dtype="bfloat16",
        )
        return llm
    elif model_name == "molmo_D":
        llm = LLM(
            model="allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            dtype="float16",
        )
        return llm
    elif model_name == "llava":
        llm = LLM(model="llava-hf/llava-1.5-7b-hf",
            max_model_len=4096,
        )
        return llm
    elif model_name == "blip2":
        llm = LLM(model="Salesforce/blip2-opt-6.7b",
        )
        return llm
    msg = f"model {model_name} is not supported."
    raise ValueError(msg)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def load_gqa_data(cfg, start_idx, end_idx):
    df = load_csv(cfg.paths.csv_path)
    # unique_image_ids = df['image_id'].unique()[:cfg.data.num_unique_images]
    unique_image_ids = df['image_id'].unique()
    # print("unique_images", len(df['image_id'].unique()))
    filtered_df = df[df['image_id'].isin(unique_image_ids)]
    filtered_df = filtered_df[filtered_df['ground_truth'].str.lower().isin(['yes', 'no'])]

    questions = []
    # create the prompts
    
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        if start_idx <= i < end_idx:
            prompt = f"Descrption: {row['scene_description']} Question: {row['question']} Answer:"
            questions.append(prompt)
        elif i >= end_idx:
            break
    return questions, filtered_df[start_idx:end_idx]

# get the whole prompts, including text descriptions
def get_multi_modal_input(cfg, start_idx, end_idx):
    questions, filtered_df_chunk = load_gqa_data(cfg, start_idx, end_idx)
    return [{"question": ques} for ques in questions], filtered_df_chunk

def main(cfg):
    df = load_csv(cfg.paths.csv_path)
    # unique_image_ids = df['image_id'].unique()[:cfg.data.num_unique_images]
    unique_image_ids = df['image_id'].unique()
    filtered_df = df[df['image_id'].isin(unique_image_ids)]
    filtered_df = filtered_df[filtered_df['ground_truth'].str.lower().isin(['yes', 'no'])]
    model = cfg.model.name
    llm = get_llm(cfg, model)

    modality = "image"

    # Perform generation in chunks
    # chunk_size = cfg.model.chunk_size
    chunk_size = cfg.model.chunk_size
    outputs = []
    for start_idx in range(0, len(filtered_df), chunk_size):
        end_idx = start_idx + chunk_size
        mm_inputs, filtered_df_chunk = get_multi_modal_input(cfg, start_idx, end_idx)
        print(f"Length of mm_inputs: {len(mm_inputs)}")

        # Prepare inputs for generation
        inputs = []
        stop_token_ids = None
        for mm_input in mm_inputs:
            # data = mm_input["data"] 
            question = mm_input["question"]
            prompt, stop_token_ids = model_example_map[model](question, modality)
            input_entry = {"prompt": prompt}
            inputs.append(input_entry)
        
        print(f"Length of Inputs: {len(inputs)}")
        
        tok = llm.get_tokenizer()
        allowed_tokens = tok([" Yes", " No", "Yes", "No", " yes", " no", "yes", "no"], add_special_tokens=False).input_ids
        allowed_tokens = [t[0] for t in allowed_tokens]

         # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            max_tokens=cfg.sampling.max_tokens, 
            allowed_token_ids=allowed_tokens,
            logprobs=1,
            stop_token_ids = stop_token_ids, 
        )
        
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