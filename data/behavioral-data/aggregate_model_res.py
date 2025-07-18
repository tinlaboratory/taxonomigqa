#!/usr/bin/env python3
import glob
import os
import re
import pandas as pd

def strict_answer_match(ground_truth: str, model_output: str) -> bool:
    if model_output.strip():
        selected_output = model_output.lower().split()[0]
        return ground_truth.lower() in selected_output
    else:
        return False

# adjust this path to wherever your CSVs live
INPUT_DIR = './'
OUTPUT_CSV = 'model_inference_output.csv'

csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.csv')))
reference_df = None
all_series  = []

for csv_file in csv_files:
    base = os.path.basename(csv_file)
    stem = os.path.splitext(base)[0]

    model_name = re.sub(r'^\d+_', '', stem)
    print(f"→ Processing {base} → column “{model_name}”")

    sep = '\t' 
    df  = pd.read_csv(csv_file, sep=sep, low_memory=False)
    df  = df[df['ground_truth'].isin(['yes', 'no'])]
    df = df[df['substitution_hop'] == 0]

    # on the *first* file only, capture all of its original columns:
    if reference_df is None:
        # reset_index so it lines up with our later series
        ref = df.reset_index(drop=True).copy()
        # drop unwanted columns if they exist
        # if "language_only" and "file_name" are not in the columns, drop them
        if 'language_only' in ref.columns:
            ref.drop(columns=['language_only'], errors='ignore', inplace=True)
        if 'file_name' in ref.columns:
            ref.drop(columns=['file_name'], errors='ignore', inplace=True)

        ref.drop(columns=['prompt', 'stop_token_ids', 'raw_question', 'model_output', 'original_question', 'ground_truth_long'],
                 errors='ignore', inplace=True)
        reference_df = ref
    # compute the strict_eval series
    strict_eval = df.apply(
        lambda row: strict_answer_match(str(row['ground_truth']),
                                        str(row['model_output'])),
        axis=1
    ).reset_index(drop=True)
    strict_eval.name = model_name
    all_series.append(strict_eval)

# now stitch the original cols + each model's strict_eval
merged = pd.concat([reference_df] + all_series, axis=1)
def lowercase_first(s):
    return s[0].lower() + s[1:] if isinstance(s, str) and len(s) > 1 else s

# drop scene_description, original_arg column
merged = merged.drop(columns=['scene_description'])
# drop some columns

# write out
merged.to_csv(OUTPUT_CSV, sep=',', index=False)
print(f"\n Wrote merged results (with original columns) to {OUTPUT_CSV}")
