# Vision-and-Language Training Helps Deploy Taxonomic Knowledge but Does Not Fundamentally Alter It


## Requirements
```
transformers
minicons
vllm
inflect
nltk
```

## TaxonomiGQA
TaxonomiGQA is a dataset constructed on top of GQA. The input file, located at
`data/behavioral-data/model_inference_input.csv`, contains the scene descriptions, questions, and target arguments used for model inference.

### Configuration

Model and experiment configurations are defined in YAML files under 
`src/configs/`

### Running Inference

To run inference for a specific model, use:
```
python run_inference.py --config="src/configs/vlm_text_qwen2.5VL.yaml"
```
This script reads input from:
`data/behavioral-data/model_inference_input.csv`
and writes model outputs to:
`data/behavioral-data/vlm_text_qwen2.5VL.csv`.
Each model will produce a separate CSV file named after its config.

### Aggregated Results

After running inference with all desired models, the individual outputs can be aggregated.
The aggregated results (across multiple models) are stored in:
`data/behavioral-data/model_inference_output.csv`

## TAXOMPS

Generate stimuli using:

```bash
python src/taxomps-computemax-stimuli.py
```

Run models using:
```bash
bash scripts/taxomps.sh
```

This script saves results in the following directories:
* `data/results/taxomps-hypernym-qa` -- for hypernyms (positive samples)
* `data/results/taxomps-ns-all-qa` -- negative samples
* `data/swapped/taxomps-swapped-qa` -- for cases where we swap hypernym and hyponym (unused in paper).

To get plots, run the following R script: `analysis/gqa-taxomps-analysis.R`


## RSA Analysis

The following script runs the Park et al., method and saves results in:
* `data/results/pair-rsa.csv` -- for RSA metrics
* `data/reps/<modelname>/long-mats/` -- for pairwise similarities

```bash
bash scripts/rsa.sh
```

To get plots, run the following R scripts: 
* `analysis/rsa-plots.R`-- for matrices
* `analysis/rsa-analysis.R` for tests

## Embedding Analysis:

The following runs the embedding similarity analysis:
```bash
python src/embedding_analysis/embedding_similarity.py \
  --emb_unemb emb \
  --results_dir data/results/embedding_analysis/
```

## Contextualized Representational Similarity Analysis

Get Qwen2.5 data by running `analysis/everything-qwen.R`, then run:

```bash
bash scripts/cwe-sims.sh
```

This will save results in `data/results/gqa-cwe-sims-all/<modelname>`

To get plots, use the following R script: `analysis/token-sim-analysis-qwen-all-no.R`

## PCA

Data used: same as previous section (Contextualized Representation Similarity) but now for PCA. 

Run `src/pca-interactive.ipynb` to run exps and save data.

Then, run `analysis/pca-attempt.R` to get pca plots.


## Image Similarity Analysis

To compute visual similarity between taxonomy nodes using Qwen2.5-VL 

```bash
cd src/similarity_analysis/code/
python compute_taxonomy_sims_image.py \
  --nonleaf_out_pkl ../data/qwen_nl_node_to_embeds.pkl \
  --leaf_out_pkl ../data/qwen_leaf_node_to_embeds.pkl \
  --sim_csv_out ../data/qwen_substituted_edge_accuracy.csv \
  --model Qwen \
  --model_type vlm-text
```

#### Arguments

* `--nonleaf_out_pkl`: Path to save or load non-leaf node image embeddings (as a pickle file).
* `--leaf_out_pkl`: Path to save or load leaf node image embeddings (as a pickle file).
* `--sim_csv_out`: Output CSV file to store similarity scores between concept pairs.
* `--model`: Name of the model used (e.g., `Qwen`, `llava`).
* `--model_type`: Type of model (e.g., `vlm-text`) used for filtering concept pairs.

#### Input Data

* **Taxonomy**: `../data/arg_hypernyms.json` – maps leaf concepts to their ancestors.
* **Annotations**: `../data/combined.json` – maps concepts to THINGS image folders.
* **Images**: Located under `../data/THINGS/object_images/`.
* **Concept Pairs**: `../data/model_substituted_edge_accuracy_with_vlm.csv` – includes model accuracies for concept pairs.

#### Output

* Embeddings for each concept (leaf and non-leaf) saved as pickle files.
* CSV file with computed cosine similarity scores between concept pairs.

To generate plots and run statistical analysis, use:

```R
analysis/viz-sim.R
```


## Citation

If you use the code in this work or use our results, please cite us using:

```bibtex
TBD
```
