# declare -a models=(lmsys/vicuna-7b-v1.5 meta-llama/Llama-3.1-8B HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-1.7B allenai/Molmo-7B-D-0924 lmsys/vicuna-7b-v1.5)

declare -a models=(allenai/Molmo-7B-D-0924 lmsys/vicuna-7b-v1.5)



for model in "${models[@]}"; do
    # python src/logprob.py --model $model\
    #     --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs.csv \
    #     --output_dir data/results/hypernym-minimal-pairs \
    #     --batch_size 16 \
    #     --device cuda:0

    # python src/taxonomic-yesno.py --model $model\
    #     --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv \
    #     --output_dir data/results/hypernym-minimal-pairs-qa \
    #     --batch_size 16 \
    #     --device cuda:0

    python src/taxonomic-yesno.py --model $model\
        --eval_path data/things-taxonomic-sensitivity/taxomps-ns-qa.csv \
        --output_dir data/results/taxomps-ns-qa \
        --batch_size 16 \
        --device cuda:0 

    python src/taxonomic-yesno.py --model $model\
        --eval_path data/things-taxonomic-sensitivity/taxomps-swapped-qa.csv \
        --output_dir data/results/taxomps-swapped-qa \
        --batch_size 16 \
        --device cuda:0 
done

# declare -a models=(Qwen/Qwen2-7B meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2 Qwen/Qwen2.5-7B-Instruct)

# for model in "${models[@]}"; do
#     # python src/logprob.py --model $model\
#     #     --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs.csv \
#     #     --output_dir data/results/hypernym-minimal-pairs \
#     #     --batch_size 16 \
#     #     --device cuda:0

#     # python src/taxonomic-yesno.py --model $model\
#     #     --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv \
#     #     --output_dir data/results/hypernym-minimal-pairs-qa \
#     #     --batch_size 16 \
#     #     --device cuda:0

#     python src/taxonomic-yesno.py --model $model\
#         --eval_path data/things-taxonomic-sensitivity/taxomps-ns-qa.csv \
#         --output_dir data/results/taxomps-ns-qa \
#         --batch_size 16 \
#         --device cuda:0 \
#         --instruct

#     python src/taxonomic-yesno.py --model $model\
#         --eval_path data/things-taxonomic-sensitivity/taxomps-swapped-qa.csv \
#         --output_dir data/results/taxomps-swapped-qa \
#         --batch_size 16 \
#         --device cuda:0 \
#         --instruct
# done

# # models that require VLMScorer
# declare -a models=(llava-hf/llava-1.5-7b-hf llava-hf/llava-onevision-qwen2-7b-ov-hf llava-hf/llava-v1.6-mistral-7b-hf Qwen/Qwen2.5-VL-7B-Instruct meta-llama/Llama-3.2-11B-Vision meta-llama/Llama-3.2-11B-Vision-Instruct)

# for model in "${models[@]}"; do
#     # python src/logprob.py --model $model\
#     #     --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs.csv \
#     #     --output_dir data/results/hypernym-minimal-pairs \
#     #     --batch_size 16 \
#     #     --device cuda:0 \
#     #     --vlmscorer

#     # python src/taxonomic-yesno.py --model $model\
#     #     --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv \
#     #     --output_dir data/results/hypernym-minimal-pairs-qa \
#     #     --batch_size 8 \
#     #     --device cuda:0 \
#     #     --vlmscorer

#     python src/taxonomic-yesno.py --model $model\
#         --eval_path data/things-taxonomic-sensitivity/taxomps-ns-qa.csv \
#         --output_dir data/results/taxomps-ns-qa \
#         --batch_size 8 \
#         --device cuda:0 \
#         --instruct \
#         --vlmscorer

#     python src/taxonomic-yesno.py --model $model\
#         --eval_path data/things-taxonomic-sensitivity/taxomps-swapped-qa.csv \
#         --output_dir data/results/taxomps-swapped-qa \
#         --batch_size 8 \
#         --device cuda:0 \
#         --instruct \
#         --vlmscorer
# done

declare -a models=(HuggingFaceTB/SmolVLM-500M-Base HuggingFaceTB/SmolVLM-256M-Base HuggingFaceTB/SmolVLM-Base)

for model in "${models[@]}"; do
    # python src/logprob.py --model $model\
    #     --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs.csv \
    #     --output_dir data/results/hypernym-minimal-pairs \
    #     --batch_size 16 \
    #     --device cuda:0 \
    #     --vlmscorer

    # python src/taxonomic-yesno.py --model $model\
    #     --eval_path data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv \
    #     --output_dir data/results/hypernym-minimal-pairs-qa \
    #     --batch_size 8 \
    #     --device cuda:0 \
    #     --vlmscorer

    python src/taxonomic-yesno.py --model $model\
        --eval_path data/things-taxonomic-sensitivity/taxomps-ns-qa.csv \
        --output_dir data/results/taxomps-ns-qa \
        --batch_size 8 \
        --device cuda:0 \
        --vlmscorer

    python src/taxonomic-yesno.py --model $model\
        --eval_path data/things-taxonomic-sensitivity/taxomps-swapped-qa.csv \
        --output_dir data/results/taxomps-swapped-qa \
        --batch_size 8 \
        --device cuda:0 \
        --vlmscorer
done

# # sims
# declare -a models=(lmsys/vicuna-7b-v1.5 meta-llama/Llama-3.1-8B)
# for model in "${models[@]}"; do
#     python src/taxonomic-sims.py --model $model\
#         --batch_size 16 \
#         --device cuda:0
# done

# declare -a models=(llava-hf/llava-1.5-7b-hf meta-llama/Llama-3.2-11B-Vision)
# for model in "${models[@]}"; do
#     python src/taxonomic-sims.py --model $model\
#         --batch_size 8 \
#         --device cuda:0 \
#         --vlm
# done
