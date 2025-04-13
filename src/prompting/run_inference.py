

import logging
import os

import pandas as pd
from datasets import Image as HFImage  # Rename to avoid clash with PIL.Image
from datasets import load_dataset
from omegaconf import OmegaConf
from PIL import Image
from vllm import LLM, SamplingParams
# ImageAsset might not be needed if passing PIL directly
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LLama 3.2
def run_mllama(question: str, modality: str):
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    if modality == "text":
        prompt = f"<|begin_of_text|>{question}"
    else:
        prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return prompt, stop_token_ids

# Molmo
def run_molmo_D(question: str, modality: str):
    prompt = question
    stop_token_ids = None
    return prompt, stop_token_ids

def run_molmo_O(question: str, modality: str):
    prompt = question
    stop_token_ids = None
    return prompt, stop_token_ids

# LLaVA-1.5
def run_llava(question: str, modality: str):
    if modality == "text":
        prompt = f"USER: \n{question}\nASSISTANT:"
    else:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
    stop_token_ids = None
    return prompt, stop_token_ids

# BLIP-2
def run_blip2(question: str, modality: str):
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

def get_llm(model_name, modality: str)->LLM:
    if modality == "text":
        # For text-only models, we can use the default LLM class
        llm = LLM(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
            dtype="float16",
        )
        return llm
    elif model_name == "mllama":
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

def load_and_prepare_dataset(cfg):
    """Loads dataset from Hugging Face Hub, filters, and prepares images."""
    repo_id = cfg.dataset.repo_id
    logging.info(f"Loading dataset from Hugging Face Hub: {repo_id}")
    try:
        # Load the dataset 
        ds = load_dataset(repo_id, split='train') 
        # If debug then take a small sample
        if cfg.processing.debug:
            logging.info("Debug mode enabled. Taking a small sample of the dataset.")
            ds = ds.select(range(50))

        # --- Apply Filtering ---
        # 1. Filter by unique images (optional, if not already done in uploaded dataset)
        num_unique_images = cfg.data.get("num_unique_images", None) 
        if num_unique_images is not None and 'image_id' in ds.column_names:
            logging.info(f"Filtering dataset to first {num_unique_images} unique image IDs.")
            unique_image_ids = ds.unique('image_id')[:num_unique_images]
            ds = ds.filter(lambda example: example['image_id'] in unique_image_ids)
            logging.info(f"Dataset size after image ID filtering: {len(ds)}")

        # 2. Filter by ground truth answer (e.g., only yes/no questions)
        valid_answers = cfg.data.get("valid_answers", None) # e.g., ['yes', 'no'] in config
        if valid_answers and 'ground_truth' in ds.column_names:
            logging.info(f"Filtering dataset to answers: {valid_answers}")
            valid_answers_lower = [ans.lower() for ans in valid_answers]
            ds = ds.filter(lambda example: example['ground_truth'].lower() in valid_answers_lower)
            logging.info(f"Dataset size after answer filtering: {len(ds)}")

        return ds

    except Exception as e:
        logging.error(f"Error loading or processing dataset from {repo_id}: {e}")
        raise # Re-raise the exception to stop execution

def main(cfg):
    # --- Load and Prepare Data ---
    prepared_dataset = load_and_prepare_dataset(cfg)
    modality = cfg.data.get("modality", "image") # Default to image if not specified
    logging.info(f"Using modality: {modality}")

    if not prepared_dataset or len(prepared_dataset) == 0:
        logging.error("Dataset is empty after loading and filtering. Exiting.")
        return

    if "image" not in prepared_dataset.column_names:
         logging.error("Dataset does not contain the 'image' column after preparation. Check config and dataset structure. Exiting.")
         return
    if "question" not in prepared_dataset.column_names:
         logging.error("Dataset does not contain the 'question' column. Check dataset structure. Exiting.")
         return

    # --- Initialize vLLM ---
    logging.info(f"Initializing LLM: {cfg.model.name}")
    model = cfg.model.name
    llm = get_llm(model, modality)

    # --- Set allowed tokens ---
    tok = llm.get_tokenizer()
    allowed_tokens = tok([" Yes", " No", "Yes", "No", " yes", " no", "yes", "no"], add_special_tokens=False).input_ids
    allowed_tokens = [t[0] for t in allowed_tokens]
    logging.info(f"Allowed tokens set: {allowed_tokens}")

    # --- Set Sampling Parameters ---
    sampling_params = SamplingParams(
        temperature=cfg.sampling.get("temperature", 0.0), # Default to greedy if not set
        top_p=cfg.sampling.get("top_p", 1.0),
        max_tokens=cfg.sampling.get("max_tokens", 64),
        
    )
    sampling_params.allowed_token_ids = allowed_tokens
    sampling_params.logprobs = cfg.sampling.get("logprobs", 1) # Default to 1 if not set
    logging.info(f"Using sampling parameters: {sampling_params}")

    # --- Prepare for Inference ---
    chunk_size = cfg.processing.get("chunk_size", 10) # Batch size for inference
    results = [] 

    # --- Perform Generation in Chunks ---
    logging.info(f"Starting inference in chunks of size {chunk_size}")
    for i in range(0, len(prepared_dataset), chunk_size):
        chunk_indices = range(i, min(i + chunk_size, len(prepared_dataset)))
        batch = prepared_dataset.select(chunk_indices) # Efficiently select a batch

        logging.info(f"Processing batch {i // chunk_size + 1} (indices {chunk_indices.start}-{chunk_indices.stop -1})")

        # Prepare inputs for vLLM
        inputs_for_llm = []
        original_data_batch = [] # Keep track of original data for merging later
        for item in batch:
            # Ensure image is PIL for vLLM if needed (HFImage usually loads as PIL)
            img_data = item['image']
            if not isinstance(img_data, Image.Image):
                 # Handle cases where it might be loaded differently, though cast_column usually ensures PIL
                 logging.warning(f"Image data is not a PIL Image: {type(img_data)}.")
                 # Add conversion logic if necessary, e.g., if it's a path or bytes
                 # This shouldn't happen with the .cast_column(..., HFImage()) approach
                 continue # Or raise error
            
            if cfg.data.get("add_scene_description", False) and 'scene_description' in item:
                question = f"Description: {item['scene_description']} Question: {item['question']} Answer:"
            else:
                question = f"Question: {item['question']} Answer:" if cfg.model.get("language_only", False) else item['question']
            
            if cfg.model.get("language_only", False):
                prompt, stop_token_ids = question, None
            else:
               prompt, stop_token_ids = model_example_map[model](question, modality)
            sampling_params.stop_token_ids = stop_token_ids
            input_entry = {
                # Adjust prompt format as needed for your specific model
                "prompt": prompt,
            }
            if modality == "image":
                input_entry["multi_modal_data"] = {"image": img_data} # Pass the PIL image object directly
            inputs_for_llm.append(input_entry)
            # Store original data point along with any necessary identifiers
            original_data_batch.append({k: v for k, v in item.items() if k != 'image'}) # Exclude bulky image data

        if not inputs_for_llm:
            logging.warning(f"Skipping empty batch {i // chunk_size + 1}")
            continue

        # Run vLLM generation
        try:
            chunk_outputs = llm.generate(inputs_for_llm, sampling_params=sampling_params)
        except Exception as e:
            logging.error(f"Error during vLLM generation for batch {i // chunk_size + 1}: {e}")
            # Decide how to handle errors: skip batch, retry, exit?
            # For now, let's store None and continue
            chunk_outputs = [None] * len(inputs_for_llm)


        # --- Process and Store Results ---
        for idx, output in enumerate(chunk_outputs):
            result_entry = original_data_batch[idx].copy() # Start with original metadata
            if output:
                 generated_text = output.outputs[0].text.strip()
                 result_entry["model_output"] = generated_text
                 # Can also store prompt, tokens used, etc. if needed
                 # result_entry["prompt_used"] = output.prompt
                 # result_entry["output_tokens"] = len(output.outputs[0].token_ids)
            else:
                 result_entry["model_output"] = "GENERATION_ERROR" # Placeholder for errors

            results.append(result_entry)

    logging.info(f"Inference complete. Processed {len(results)} samples.")

    # --- Save Results ---
    if results:
        output_df = pd.DataFrame(results)
        output_path = cfg.paths.get("output_path", "output.csv") 
        # Adjust separator if needed (e.g., '\t')
        output_sep = cfg.paths.get("output_separator", ",")
        logging.info(f"Saving results to {output_path}")
        try:
             output_df.to_csv(output_path, sep=output_sep, index=False)
             logging.info("Results saved successfully.")
        except Exception as e:
             logging.error(f"Failed to save results to {output_path}: {e}")
    else:
        logging.warning("No results were generated to save.")
        

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description='Using vLLM for testing/inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    main(cfg)