# this file is to build a dataframe that stores all the necessary information for prompting and evaluation
import json
import pandas as pd

def load_json(path:str):
    with open(path, 'r') as f:
        return json.load(f)
    
# question id start from 0 
if __name__ == "__main__":
    # load scene descriptions as list of strings
    ROOT_PATH = "/projectnb/tin-lab/yuluq/data/subset_combined_stats_data/test/"
    scene_text_path = ROOT_PATH + "val_scene_to_text.json"
    val_q_path = ROOT_PATH + "val_q_combined.json"
    
    text_descriptions = load_json(scene_text_path)
    for key, val in text_descriptions.items():
        text_descriptions[key] = " ".join(val) 
    val_qs = load_json(val_q_path)

    rows = []
    question_id = 0
    for i, (key, val) in enumerate(val_qs.items()):
        for entry in val:
            image_id = key
            original_question_id = None
            question = ""
            scene_description = text_descriptions[key]
            questions = entry['questions']
            ground_truth = entry['answer']
            ground_truth_long = entry['fullAnswer']
            for k, item in enumerate(questions):
                question = questions[k]
                question_type = k
                question_id += 1
                dict_to_be_added = {
                    "question_id": question_id, 
                    "image_id": image_id, 
                    "original_question_id": original_question_id,
                    "question" : question, 
                    "question_type": k,
                    "scene_description": scene_description, 
                    "ground_truth": ground_truth,
                    "ground_truth_long": ground_truth_long,
                }
                if k == 0:
                    original_question_id = int(question_id)

                rows.append(dict_to_be_added)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT_PATH + "output.tsv", sep='\t', index=False)
    
                

                
            


                    









