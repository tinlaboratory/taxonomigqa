# multimodal-representations

## Requirements
```
transformers
minicons
vllm
inflect
nltk
```


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

TBD

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

To run IMG-sim, run: TBD

The data is saved in: TBD

For statistical analysis and plots, run `analysis/viz-sim.R`


## Citation

If you use the code in this work or use our results, please cite us using:

```bibtex
TBD - Qin et al., 2025
```
