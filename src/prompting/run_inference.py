from vllm import LLM, SamplingParams
# ImageAsset might not be needed if passing PIL directly
from vllm.assets.image import ImageAsset
from vllm.utils import FlexibleArgumentParser
from PIL import Image
import os
import pandas as pd
from omegaconf import OmegaConf
from datasets import load_dataset, Image as HFImage # Rename to avoid clash with PIL.Image
import logging
import time 
from functools import partial
from transformers import AutoTokenizer 
import multiprocessing
# Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

import os
from datetime import datetime

########### Logging Setup ###########
def set_up_logger(cfg):
    log_dir = "../model_logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    model_name = cfg.model.name.replace("/", "_")
    
    log_path = os.path.join(log_dir, f"{model_name}__{ts}.log")

    logging.basicConfig(
        filename=log_path,         # ← write log lines here
        filemode='w',              # ← 'w' to overwrite, 'a' to append
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Logging to {log_path}")

################################################################

# LLama 3.2
def run_mllama(question: str, language_only: bool, tokenizer):
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    if language_only:
        prompt = f"<|begin_of_text|>{question}"
    else:
        prompt = f"<|image|><|begin_of_text|>{question} Answer:"
    stop_token_ids = None
    return prompt, stop_token_ids

# Molmo
def run_molmo_D(question: str, language_only: bool, tokenizer):
    if not language_only:
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> \
        <|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{question}<|im_end|> \
        <|im_start|>assistant\n"
    stop_token_ids = None
    return prompt, stop_token_ids

# LLaVA-1.5
def run_llava(question: str, language_only: bool, tokenizer):
    if language_only:
        prompt = f"USER: \n{question}\nASSISTANT:"
    else:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
    stop_token_ids = None
    return prompt, stop_token_ids

def run_llava_next(question: str, language_only: bool, tokenizer):
    if language_only:
        prompt = f"[INST] {question} [/INST]"
    else:
        prompt = f"[INST] <image>\n{question} [/INST]"
    stop_token_ids = None
    return prompt, stop_token_ids

def run_llava_onevision(question: str, language_only: bool, tokenizer):
    if language_only:
        prompt = f"<|im_start|>user\n{question}<|im_end|> <|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user <image>\n{question}<|im_end|> <|im_start|>assistant\n"
    stop_token_ids = None
    return prompt, stop_token_ids

def run_qwen2_5_vl(question: str, language_only: bool, tokenizer):
    if language_only:
        vision_part = ""
    else:
        vision_part = "<|vision_start|><|image_pad|><|vision_end|>" 

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{vision_part}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    stop_token_ids = None
    return prompt, stop_token_ids

def helper_func(model_name: str, language_only: bool, question: str, tokenizer):
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    content = question if language_only else f"<image>\n{question}"
    messages = [{'role': 'user', 'content': content}]
    prompt = tokenizer.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)[0]

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens if tokenizer.convert_tokens_to_ids(t) is not None]
    return prompt, stop_token_ids

def run_internvl_3(question: str, language_only: bool, tokenizer):
    
    model_name = "OpenGVLab/InternVL3-8B"
    prompt, stop_token_ids = helper_func(model_name, language_only, question, tokenizer)

    return prompt, stop_token_ids

def run_internvl_2_5(question: str, language_only: bool, tokenizer):
    model_name = "OpenGVLab/InternVL2_5-8B"
    prompt, stop_token_ids = helper_func(model_name, language_only, question, tokenizer)

    return prompt, stop_token_ids
def run_mllama_instruct(question: list[str], language_only: bool, tokenizer):
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.

    content = [{"type": "image"}, {"type": "text","text": question}] if not language_only else [{"type": "text","text": question}]
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{
        "role": "user",
        "content": content
    }] 
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    stop_token_ids = None
    return prompt, stop_token_ids

model_example_map = {
    "llava": run_llava,
    "llava_ov": run_llava_onevision, 
    "llava_next": run_llava_next, 
    "mllama": run_mllama,
    "molmo_D": run_molmo_D,
    "qwen2.5VL": run_qwen2_5_vl,
    "InternVL3": run_internvl_3,
    "InternVL2.5": run_internvl_2_5, 
    "mllama_instruct": run_mllama_instruct
}

def get_llm(model_name, modality: str)->LLM:
    if modality == "text":
        # For text-only models, we can use the default LLM class
        llm = LLM(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        return llm
    elif model_name == "mllama":
        llm = LLM(
            model="meta-llama/Llama-3.2-11B-Vision",
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
            dtype="float16",
            limit_mm_per_prompt={"image": 1},
        )
        return llm
    elif model_name == "molmo_D":
        llm = LLM(
            model="allenai/Molmo-7B-D-0924",
            trust_remote_code=True,
            dtype="float16",
            limit_mm_per_prompt={"image": 1},
        )
        return llm
        
    elif model_name == "llava":
        llm = LLM(
            model="llava-hf/llava-1.5-7b-hf",
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
        )
        return llm
    elif model_name == "qwen2.5VL":
        llm = LLM(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            max_model_len=4096,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            },
            limit_mm_per_prompt={"image": 1},
        )
        return llm 
    elif model_name == "llava_next":
        llm = LLM(
            model="llava-hf/llava-v1.6-mistral-7b-hf",
            max_model_len=8192,
            limit_mm_per_prompt={"image": 1},
        )
        return llm 
    elif model_name == "InternVL2.5": 
        llm = LLM(
            model="OpenGVLab/InternVL2_5-8B",
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
        )
        return llm 
    elif model_name == "InternVL3": 
        llm = LLM(
            model="OpenGVLab/InternVL3-8B",
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
        )
        return llm 
    elif model_name == "llava_ov":
        llm = LLM(
            model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
            max_model_len=16384,
            limit_mm_per_prompt={"image": 1},
        )
        return llm 
    elif model_name == "mllama_instruct":
        llm = LLM(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct",
            max_model_len=2048,
            max_num_seqs=8,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 1},
        )
        return llm 

    msg = f"model {model_name} is not supported."
    raise ValueError(msg)

# def attach_cached_image(example, image_cache):
#     filename = os.path.basename(example["file_name"])
#     return {"image": image_cache.get(filename)}

import sys
def timed(fn):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f'Function {fn.__name__!r} executed in {endt - startt:.3f}s', file=sys.stderr)
        return result
    return wrapper

@timed
def apply_chat_template_lm(question, model_name, tokenizer):
    if getattr(tokenizer, "chat_template", None):
        # Now it's safe to call apply_chat_template
        messages = [
                {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = question
    stop_token_ids = None

    return prompt, stop_token_ids

def load_and_prepare_dataset(cfg):
    """Loads dataset from Hugging Face Hub, filters, and prepares images."""
    repo_id = cfg.dataset.repo_id
    logging.info(f"Loading dataset from Hugging Face Hub: {repo_id}")
    try:
        # Load the dataset

        ds = load_dataset(repo_id, "text_only" if cfg.model.language_only else "image_text", split="train", trust_remote_code=True) 
        
        # valid_answers = cfg.data.get("valid_answers", ["yes", "no"]) # e.g., ['yes', 'no'] in config
        # if valid_answers and 'ground_truth' in ds.column_names:
        #     logging.info(f"Filtering dataset to answers: {valid_answers}")
        #     valid_answers_lower = [ans.lower() for ans in valid_answers]
        #     ds = ds.filter(lambda example: example['ground_truth'].lower() in valid_answers_lower)
        #     logging.info(f"Dataset size after answer filtering: {len(ds)}")
        
          # If debug then take a small sample
        if cfg.processing.debug:
            logging.info("Debug mode enabled. Taking a small sample of the dataset.")
            ds = ds.select(range(50))

        return ds

    except Exception as e:
        logging.error(f"Error loading or processing dataset from {repo_id}: {e}")
        raise # Re-raise the exception to stop execution

def build_prompt(item, cfg, tokenizer):
    if cfg.data.get("add_scene_description", False) and 'scene_description' in item:
        raw_question = item['question']
        # lowercase the first letter of the question
        if raw_question[0].isupper():
            raw_question = raw_question[0].lower() + raw_question[1:]
        question = f"Description: {item['scene_description']} Question: In the scene, {raw_question} Answer:"
    else:
        question = f"Question: {item['question']} Answer:" if cfg.model.get("language_only") else item['question']

    if cfg.data.get("modality", "text") == "text":
        prompt, stop_token_ids = apply_chat_template_lm(question, cfg.model.name, tokenizer)
    else:
        prompt, stop_token_ids = model_example_map[cfg.model.name](question, cfg.model.get("language_only", True), tokenizer)
    
    return prompt, stop_token_ids, question


def main(cfg):
    # --- Load and Prepare Data ---
    # print one prompt
    print_prompt = True
    prepared_dataset = load_and_prepare_dataset(cfg)
    modality = cfg.data.get("modality", "image") # Default to image if not specified
    logging.info(f"Using modality: {modality}")

    if not prepared_dataset or len(prepared_dataset) == 0:
        logging.error("Dataset is empty after loading and filtering. Exiting.")
        return

    if cfg.data.modality == 'image' and not cfg.model.language_only and "image" not in prepared_dataset.column_names:
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

    start = time.time()
    # Precompute prompts before batching
    prepared_dataset = prepared_dataset.map(
        lambda item: {
            **item,
            **dict(zip(["prompt", "stop_token_ids", "raw_question"], build_prompt(item, cfg, tok))),
        },
        num_proc=cfg.processing.get("num_proc", 4), # Number of processes for parallel processing
    )

    print (f" I took {time.time() - start} seconds to prepare the dataset")
    

    # --- Perform Generation in Chunks ---
    prepared_dataset = list(prepared_dataset) # Preload the dataset into memory for efficient access
    logging.info(f"Starting inference in chunks of size {chunk_size}")
    for i in range(0, len(prepared_dataset), chunk_size):
        start = time.time()
        chunk_indices = slice(i, min(i + chunk_size, len(prepared_dataset)))
        batch = prepared_dataset[chunk_indices] # Efficiently select a batch
        print(f" I took {time.time() - start} seconds to load the batch")

        logging.info(f"Processing batch {i // chunk_size + 1} (indices {chunk_indices.start}-{chunk_indices.stop -1})")

        # Prepare inputs for vLLM
        inputs_for_llm = []
        original_data_batch = [] # Keep track of original data for merging later
        for item in batch:
            # Ensure image is PIL for vLLM if needed (HFImage usually loads as PIL)
            if cfg.data.get("modality", "image") == "image" and not cfg.model.language_only:
                img_data = item['image']

            sampling_params.stop_token_ids = item["stop_token_ids"]
            input_entry = {
                # Adjust prompt format as needed for your specific model
                "prompt": item["prompt"],
            }
            # only print one prompt for sanity check 
            if print_prompt:
                prmpt = item["prompt"]
                logging.info(f"Example Prompt for {cfg.model.name}: {prmpt}")
                print_prompt = False

            if modality == "image" and not cfg.model.language_only:
                input_entry["multi_modal_data"] = {"image": img_data} # Pass the PIL image object directly
            inputs_for_llm.append(input_entry)
            # Store original data point along with any necessary identifiers
            original_data_batch.append({k: v for k, v in item.items() if k != 'image'}) # Exclude bulky image data

        if not inputs_for_llm:
            logging.warning(f"Skipping empty batch {i // chunk_size + 1}")
            continue

        print(f" I took {time.time() - start} seconds to prepare the batch")
        # Run vLLM generation
        try:
            chunk_outputs = llm.generate(inputs_for_llm, sampling_params=sampling_params)
        except Exception as e:
            logging.error(f"Error during vLLM generation for batch {i // chunk_size + 1}: {e}")
            chunk_outputs = [None] * len(inputs_for_llm)

        post_generation_time = time.time()
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

        print(f" I took {time.time() - post_generation_time} seconds to process the batch")

    logging.info(f"Inference complete. Processed {len(results)} samples.")

    # --- Save Results ---
    if results:
        output_df = pd.DataFrame(results)
        output_path = cfg.paths.get("output_path", "output.csv") 
        # Adjust separator if needed (e.g., '\t')
        output_sep = cfg.paths.get("output_separator", "\t")
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
    
    # start timing 
    start_time = time.time()
    # Load config
    cfg = OmegaConf.load(args.config)
    set_up_logger(cfg)
    main(cfg)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")
    logging.info(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")