# declare -a models=(google/gemma-2-9b-it google/gemma-2-9b)

# for model in "${models[@]}"; do
#     python src/tsv.py --model $model
#     python src/tsv_qa.py --model $model --tsv_stimuli data/things/tsv/stimuli/things-tsv-qa-stimuli.csv
#     python src/tsv_qa.py --model $model --tsv_stimuli data/things/tsv/stimuli/things-tsv-qa-declarative-stimuli.csv
# done

python src/tsv.py --model google/gemma-2-9b-it --chat --batch_size 32
# python src/tsv.py --model google/gemma-2-9b --batch_size 32
