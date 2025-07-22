declare -a datasets=(hypernym swapped ns-all)

# declare -a models=(HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-1.7B)

# for model in "${models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#     echo $model $dataset
#         python src/taxomps-minimal.py --model $model\
#             --eval_path data/gqa_entities/taxomps-$dataset.csv \
#             --output_dir data/results/smollm/taxomps-$dataset-qa \
#             --batch_size 16 \
#             --device cuda:0 
#     done
# done

declare -a models=(HuggingFaceTB/SmolLM2-135M-Instruct HuggingFaceTB/SmolLM2-360M-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo $model $dataset
        python src/taxomps-minimal.py --model $model\
            --eval_path data/gqa_entities/taxomps-$dataset.csv \
            --output_dir data/results/smollm/taxomps-$dataset-qa \
            --batch_size 16 \
            --device cuda:0 \
            --instruct
    done
done

# declare -a models=(HuggingFaceTB/SmolVLM-500M-Base HuggingFaceTB/SmolVLM-256M-Base HuggingFaceTB/SmolVLM-Base HuggingFaceTB/SmolVLM-Instruct HuggingFaceTB/SmolVLM-256M-Instruct HuggingFaceTB/SmolVLM-500M-Instruct)

# for model in "${models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         echo $model $dataset
#         python src/taxomps-minimal.py --model $model\
#             --eval_path data/gqa_entities/taxomps-$dataset.csv \
#             --output_dir data/results/smollm/taxomps-$dataset-qa \
#             --batch_size 16 \
#             --device cuda:0 \
#             --instruct \
#             --vlmscorer
#     done
# done