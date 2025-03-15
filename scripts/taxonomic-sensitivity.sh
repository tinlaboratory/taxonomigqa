declare -a models=(allenai/Molmo-7B-D-0924 Qwen/Qwen2-7B lmsys/vicuna-7b-v1.5 meta-llama/Llama-3.1-8B)

for model in "${models[@]}"; do
    python src/logprob.py --model $model\
        --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs.csv \
        --output_dir data/results/hypernym-minimal-pairs \
        --batch_size 16 \
        --device cuda:0
done

# models that require VLMScorer
declare -a models=(llava-hf/llava-1.5-7b-hf meta-llama/Llama-3.2-11B-Vision)

for model in "${models[@]}"; do
    python src/logprob.py --model $model\
        --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs.csv \
        --output_dir data/results/hypernym-minimal-pairs \
        --batch_size 16 \
        --device cuda:0 \
        --vlmscorer
done