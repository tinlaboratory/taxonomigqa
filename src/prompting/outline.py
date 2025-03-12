

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
import os
import pandas as pd
from omegaconf import OmegaConf
import outlines
from outlines import models
from constant import REGEX_EXPR
from transformers import AutoTokenizer


# class VLLMWrapper:
#     def __init__(self, model_name, llm):
#         self.llm = llm
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)

#     def generate(self, prompt, **kwargs):
#         return self.llm.generate(prompt, **kwargs)

# LLama 3.2
def run_mllama(question: str, modality: str):
    assert modality == "image"

    # model_name = "meta-llama/Llama-3.2-11B-Vision"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    
    prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return prompt, stop_token_ids

# LLaVA-1.5
def run_llava(question: str, modality: str):
    assert modality == "image"

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    stop_token_ids = None
    return prompt, stop_token_ids

model_example_map = {
    "mllama": run_mllama,
    "llava": run_llava,
}

def get_llm(cfg, model_name):
    if model_name == "mllama":
        llm = models.vllm(
            model_name="meta-llama/Llama-3.2-11B-Vision",
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
            dtype="float16",
        )
        # vllm_wrapper = VLLMWrapper(model_name="meta-llama/Llama-3.2-11B-Vision", llm=llm)
        return llm
    elif model_name == "llava":
        llm = models.vllm(model_name="llava-hf/llava-1.5-7b-hf",
            max_model_len=4096,
        )
        # vllm_wrapper = VLLMWrapper(model_name="meta-llama/Llama-3.2-11B-Vision", llm=llm)
        # print(vllm_wrapper.tokenizer)
        # return vllm_wrapper
        return llm
    
    msg = f"model {model_name} is not supported."
    raise ValueError(msg)

def load_csv(path: str) -> pd.DataFrame:
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
    model = cfg.model.name
    llm = get_llm(cfg, model)
    modality = "image"
    
    # Perform generation in chunks
    chunk_size = 1000
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
            prompt, stop_token_ids = model_example_map[model](question, modality)
            input_entry = {
                "prompt": prompt, 
                "multi_modal_data": {
                    modality: data
                },
            }
            inputs.append(input_entry)

            # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0,
            top_p=cfg.sampling.top_p,
            max_tokens=cfg.sampling.max_tokens,
            stop_token_ids=stop_token_ids, 
        )
        print(f"Length of Inputs: {len(inputs)}")
        
        # generator = outlines.generate.regex(
        # llm, REGEX_EXPR, sampler=outlines.samplers.GreedySampler())
        generator = outlines.generate.choice(llm, ["Ġyes","Ġyes", "ĠNo", "Ġno", 'yes', 'no', "Yes", "No", ' yes', ' no', ' Yes', ' No'])
        chunk_outputs = generator(inputs, sampling_params=sampling_params)
        outputs.extend(chunk_outputs)
        print(f"outputs: {len(outputs)}, {outputs[0]}")
        
        for i, (index, row) in enumerate(filtered_df_chunk.iterrows()):
            if i < len(chunk_outputs):
                # generated_text = chunk_outputs[i].outputs[0].text
                generated_text = chunk_outputs[0]
                filtered_df.at[index, "model_output"] = generated_text
    
    filtered_df.to_csv(cfg.paths.output_path, sep='\t')

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Using outlines for Llama 3.2 Vision inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    main(cfg)





