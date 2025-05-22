
declare -a models=(allenai/Molmo-7B-D-0924 Qwen/Qwen2-7B Qwen/Qwen2-7B-Instruct lmsys/vicuna-7b-v1.5 meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2 Qwen/Qwen2.5-7B-Instruct HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-1.7B)

for model in "${models[@]}"; do
    # python get_wordnet_hypernyms.py --model $model

    python src/rsa/compute_similarities.py --model $model
    # echo Done with $model!
done

# VLMs

declare -a models=(llava-hf/llava-1.5-7b-hf llava-hf/llava-onevision-qwen2-7b-ov-hf llava-hf/llava-v1.6-mistral-7b-hf Qwen/Qwen2.5-VL-7B-Instruct meta-llama/Llama-3.2-11B-Vision meta-llama/Llama-3.2-11B-Vision-Instruct HuggingFaceTB/SmolVLM-500M-Base HuggingFaceTB/SmolVLM-256M-Base HuggingFaceTB/SmolVLM-Base)

for model in "${models[@]}"; do
    python src/rsa/get_wordnet_hypernyms.py --model $model

    python src/rsa/compute_similarities.py --model $model --vlm
    echo Done with $model!
done

python src/rsa/rsa-heatmap.py