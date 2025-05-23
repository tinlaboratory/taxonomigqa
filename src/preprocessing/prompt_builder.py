# this file is to build a dataframe that stores all the necessary information for prompting and evaluation
import json

import pandas as pd


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# question id start from 0
if __name__ == "__main__":
    # load scene descriptions as list of strings
    ROOT_PATH = "<anonymous>/subset_combined_stats_data/test/"
    scene_text_path = ROOT_PATH + "val_scene_to_text.json"
    # val_q_path = ROOT_PATH + "val_q_combined.json"
    val_q_path = ROOT_PATH + "0311_val_question.json"

    text_descriptions = load_json(scene_text_path)
    for key, val in text_descriptions.items():
        text_descriptions[key] = " ".join(val)
    val_qs = load_json(val_q_path)

    rows = []
    question_id = None
    for i, (key, val) in enumerate(val_qs.items()):
        for entry in val:
            image_id = key
            question = ""
            scene_description = text_descriptions[key]
            args = [entry["argument"]] + entry["hypernym"]
            questions = entry["questions"]
            ground_truth = entry["answer"]
            ground_truth_long = entry["fullAnswer"]
            argument = entry["argument"]
            original_arg = entry["argument"]
            for k, (arg, item) in enumerate(zip(args, questions)):
                question_id = entry["question_id"]
                question = questions[k]
                question_type = entry["question_type"]
                substitution_hop = k
                dict_to_be_added = {
                    "question_id": question_id, 
                    "image_id": image_id, 
                    "question" : question, 
                    "original_question": questions[0],
                    "question_type": question_type,
                    "substitution_hop": substitution_hop,
                    "argument": arg,
                    "original_arg": original_arg,
                    "scene_description": scene_description,
                    "ground_truth": ground_truth,
                    "ground_truth_long": ground_truth_long,
                }

                rows.append(dict_to_be_added)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT_PATH + "0509_output.tsv", sep='\t', index=False)
    
                

                
            


                    









