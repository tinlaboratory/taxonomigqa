import argparse
import os
import sys

import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
)
import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/projectnb/tin-lab/yuluq/multimodal-representations/src/evaluation/negative_sampling/data/behavioral_test_data/negative_sampling_data_for_similarity_analsysis/",
        help="dataset name",
    )

    return parser.parse_args()


def get_hypers(hyper_tree, noun):
    return hyper_tree[noun].path()[1:-1]

# def get_hypernyms(hypernym_path):
#     hyper_tree = get_tree(hypernym_path)
#     arg_dict = {}
#     for arg in args:
#         hyp = get_hypers(hyper_tree, arg)
#         arg_dict[arg] = hyp
#     print(arg_dict)
#     print(len(arg_dict))
#     return arg_dict


def get_model_name(filename):
    if "llava" in filename:
        model = "llava"
    elif "mllama" in filename:
        model = "mllama"
    elif "molmo" in filename:
        model = "molmo"
    elif "qwen" in filename:
        model = "qwen"
    elif "vicuna" in filename:
        model = "vicuna"
    elif "llama" in filename:
        model = "llama"

    return model

def process_csv(model, model_type, val, arg_hypernyms):
    # Load the CSV file
    print(f"Processing {val}")
    # comma_sep_files = ["0426_vlm_llava.csv"]
    # sep = ',' if any(f in val for f in comma_sep_files) else '\t'
    sep = ','
    df = pd.read_csv(f"{val}",sep=sep)
    # df = pd.read_csv(f"{val}", sep="\t") if "vlm_llava" in val:
    # df['strict_eval'] = df.apply(lambda row: strict_answer_match(str(row['ground_truth']), str(row['model_output'])), axis=1)
    # args = set(df[df['substitution_hop'] == 0]['argument'].tolist())

    # start the algo
    # create a res df
    intermediate_results = {}
    for arg, hyps in arg_hypernyms.items():
        df_args = df[(df["original_arg"] == arg)]
        words = [arg] + hyps
        for i, word in enumerate(words[:-1]):
            # this version only considers the animal category... modified on 4.14
            # if hyps[-1] == "animal": # modified on 5.6
            for j in range(i+1, len(hyps)+1):
                df_child_correct = df_args[(df_args['substitution_hop'] == i) & (df_args['strict_eval'] == True)]
                uniq_ids_in_child = df_child_correct['question_id'].unique()
                df_parent_conditioned = df_args[(df_args['question_id'].isin(uniq_ids_in_child)) & (df_args['substitution_hop'] != i)]
                df_parent_correct = df_parent_conditioned[(df_parent_conditioned['substitution_hop'] == j) & (df_parent_conditioned['strict_eval'] == True)]
                if (word, words[j], hyps[-1]) not in intermediate_results:
                    if len(df_parent_correct) != 0 and len(df_child_correct) == 0:
                        print(word, words[j], hyps[-1])
                    intermediate_results[(word, words[j], hyps[-1])] = [len(df_parent_correct), len(df_child_correct)]
                else:
                    intermediate_results[(word, words[j], hyps[-1])][0] += len(df_parent_correct)
                    intermediate_results[(word, words[j], hyps[-1])][1] += len(df_child_correct)
            else:
                pass

    # concatenate intermediate results
    result_rows = []
    for (concept1, concept2, category), (parent_correct, child_correct) in intermediate_results.items():
        accuracy = parent_correct / child_correct * 100 if child_correct != 0 else 0
        new_row = {
            'model': model,
            'model-type': model_type,
            'concept1': concept1,
            'concept2': concept2,
            'raw-counts': (parent_correct, child_correct),
            'accuracy': accuracy,
            'category': category
        }
        result_rows.append(new_row)
    return pd.DataFrame(result_rows)

def get_sub_category_accuracy(df, arg_hypernyms):
    # append top category to the dataframe
    # iterate through the dataframe
    # for each row, get the top category

    for index, row in df.iterrows():
        concept1 = row['concept1']
        top_cat = arg_hypernyms[concept1][-1]
        df.at[index, 'category'] = top_cat
    # group by category
    df_grouped = df.groupby('category')
    # get the accuracy for each category
    category_accuracy = {}
    for category, group in df_grouped:
        # get the accuracy for each group
        accuracy = group['accuracy'].mean()
        category_accuracy[category] = accuracy
    
    return category_accuracy

def parse_filename(filename):
    """
    Extract model_type and model_name from the filename.
    E.g., 0426_lm_Llama_3.1_8B_Instruct.csv -> ('lm', 'Llama_3.1_8B_Instruct')
    """
    name = os.path.basename(filename).replace(".csv", "")
    parts = name.split("_")
    
    # Try matching more specific types first (e.g., lm_q_only before lm)
    for key, model_type in sorted(model_type_variants.items(), key=lambda x: -len(x[0])):
        type_parts = key.split("_")
        if parts[0:len(type_parts)] == type_parts:
            model_name = "_".join(parts[1+len(type_parts):])
            return model_type, model_name
    return None, None


if __name__ == "__main__":

    # Define the base path and hypernym path
    BASE_PATH = "/projectnb/tin-lab/yuluq/"
    hypernym_path = (
        BASE_PATH + "multimodal-representations/data/gqa_entities/noun-hypernyms.json"
    )
    arg_hyp_path = "./data/arg_hypernyms.json"
    file_name = "negative_sample_model_substitued_edge_accuracy.csv"

    # Parse args
    args = parse_args()
    data_folder = args.data_folder


    # load or create arg-hyp dict
    if os.path.exists(arg_hyp_path):
        with open(arg_hyp_path, "r") as f:
            arg_hypernyms = json.load(f)
    # else:
    #     arg_hypernyms = get_hypernyms(hypernym_path)
    #     with open(arg_hyp_path, 'w') as f:
    #         json.dump(arg_hypernyms, f)
    
    # check if data_folder}/{file_name exists
    if os.path.exists(f"./data/{file_name}"):
        # load the file
        res_df = pd.read_csv(f"{data_folder}/{file_name}", sep='\t')
    else: 
        # Obtain csv files
        csv_files = glob.glob(f"{data_folder}/*.csv")
        # vlm_text = glob.glob(f"{data_folder}/*text_only_*.csv")
        # lm = glob.glob(f"{data_folder}/*constrained_decoding_*.csv")

        model_type_variants = {
                # 'lm_q_only': 'lm-q-only',
                # 'vlm_q_only': 'vlm-q-only',
                'vlm_text': 'vlm-text',
                'lm': 'lm',
                # 'vlm': 'vlm',
            }
        
        res_df = pd.DataFrame(columns=['model', 'model-type', 'concept1', 'concept2', 'raw-counts', 'accuracy', "category"])
        results_rows = []
        for file in tqdm(csv_files, desc="Processing files"):
            model_type, model_name = parse_filename(file)
            if model_type is None:
                print(f"Skipping unrecognized file: {file}")
                continue
            append_df = process_csv(model_name, model_type, file, arg_hypernyms)
            results_rows.append(append_df)
        res_df = pd.concat(results_rows, ignore_index=True)
        # save the results
        # if os.path.exists(f"{data_folder}/edge_accuracy.csv"):
        res_df.to_csv(f"/projectnb/tin-lab/yuluq/multimodal-representations/src/evaluation/negative_sampling/data/behavioral_test_data/negative_sampling_data_for_similarity_analsysis/{file_name}", sep='\t', index=False)    
    
    # acc = get_sub_category_accuracy(res_df, arg_hypernyms)
    # # save the accuracy to a csv file 
    # acc_df = pd.DataFrame.from_dict(acc, orient='index', columns=['accuracy'])
    # acc_df.to_csv(f"{data_folder}/sub_category_accuracy.csv", sep='\t', index=True)

