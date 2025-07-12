
declare -a datasets=(hypernym swapped ns-all)
# declare -a datasets=(ns-all)

declare -a models=(lmsys/vicuna-7b-v1.5 meta-llama/Llama-3.1-8B HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-1.7B allenai/Molmo-7B-D-0924)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
    echo $model $dataset
        python src/taxomps-minimal.py --model $model\
            --eval_path data/gqa_entities/taxomps-$dataset.csv \
            --output_dir data/results/taxomps-$dataset-qa \
            --batch_size 16 \
            --device cuda:0 
    done
done


declare -a models=(Qwen/Qwen2-7B meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2-7B-Instruct)
# declare -a models=(Qwen/Qwen2-7B-Instruct)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo $model $dataset
        python src/taxomps-minimal.py --model $model\
            --eval_path data/gqa_entities/taxomps-$dataset.csv \
            --output_dir data/results/taxomps-$dataset-qa \
            --batch_size 16 \
            --device cuda:0 \
            --instruct
    done
done



declare -a models=(llava-hf/llava-1.5-7b-hf llava-hf/llava-onevision-qwen2-7b-ov-hf llava-hf/llava-v1.6-mistral-7b-hf Qwen/Qwen2.5-VL-7B-Instruct meta-llama/Llama-3.2-11B-Vision HuggingFaceTB/SmolVLM-500M-Base HuggingFaceTB/SmolVLM-256M-Base HuggingFaceTB/SmolVLM-Base)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo $model $dataset
        python src/taxomps-minimal.py --model $model\
            --eval_path data/gqa_entities/taxomps-$dataset.csv \
            --output_dir data/results/taxomps-$dataset-qa \
            --batch_size 16 \
            --device cuda:0 \
            --instruct \
            --vlmscorer
    done
done

declare -a models=(meta-llama/Llama-3.2-11B-Vision-Instruct)

for model in "${models[@]}"; do
    # echo $model
    for dataset in "${datasets[@]}"; do
        echo $model $dataset
        python src/taxomps-minimal.py --model $model\
            --eval_path data/gqa_entities/taxomps-$dataset.csv \
            --output_dir data/results/taxomps-$dataset-qa \
            --batch_size 4 \
            --device cuda:0 \
            --instruct \
            --vlmscorer
    done
done