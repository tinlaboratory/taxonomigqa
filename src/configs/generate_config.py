import datetime
import os
import yaml
from copy import deepcopy

# Basic config template
base_config = {
    "paths": {
        "root_path": "../../data/behavioral-data", 
        "output_path": ""  # Will be filled later
    },
    "dataset": {
        "repo_id": "tin-lab/TaxonomiGQA"
    },
    "model": {
        "name": "",  # Will fill
        "max_model_len": 4096,
        "mm_cache_preprocessor": True,
        "language_only": True  # Might toggle
    },
    "processing": {
        "debug": False,
        "chunk_size": 50
    },
    "sampling": {
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 1,
        "batch_size": 64
    },
    "data": {
        "num_unique_images": 2014,
        "modality": "",  # Will fill
        "add_scene_description": True  # Might toggle
    }
}

# Model lists
vlm_models = [
    "llava", "llava_ov", "llava_next", "mllama", "molmo_D",
    "qwen2.5VL", "mllama_instruct"
]

lm_models = [
    "Qwen/Qwen2-7B", "meta-llama/Llama-3.1-8B", "lmsys/vicuna-7b-v1.5",
    "mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct" #Qwen/Qwen2.5-7B
]

# Settings
vlm_settings = ["vlm", "vlm_text", "vlm_q_only"]
lm_settings = ["lm", "lm_q_only"]

# Output folder
os.makedirs("configs_generated", exist_ok=True)

# Loop and generate all configs
counter = 0
for model_name in vlm_models + lm_models:
    model_type = "VLM" if model_name in vlm_models else "LM"
    settings = vlm_settings if model_type == "VLM" else lm_settings

    for setting in settings:
        cfg = deepcopy(base_config)
        cfg["model"]["name"] = model_name
        cfg["data"]["modality"] = "image" if setting.startswith("vlm") else "text"

        # Optional: If "question-only", you probably don't want scene description
        if setting == "vlm_text" or setting == "lm":
            cfg["data"]["add_scene_description"] = True
        else:
            cfg["data"]["add_scene_description"] = False

        # Auto-set language_only
        if model_type == "LM":
            cfg["model"]["language_only"] = True
        else:
            cfg["model"]["language_only"] = ( setting != "vlm")  # For VLM, only image modality is not language_only

        # Auto-set output path
        clean_model_name = model_name.replace("/", "_").replace("-", "_")
        cfg["paths"]["output_path"] = f"{cfg['paths']['root_path']}/model_outputs/{setting}_{clean_model_name}.csv"

        filename = f"./{setting}_{clean_model_name}.yaml"
        with open(filename, "w") as f:
            yaml.dump(cfg, f)

        counter += 1

print(f"Generated {counter} config files.")
