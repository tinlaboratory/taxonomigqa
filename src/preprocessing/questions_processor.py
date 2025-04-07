# group questions by image
# count questions per image
# calculate the average
import json
import matplotlib.pyplot as plt
from collections import Counter
import os

def load_json(path:str):
    with open(path, 'r') as f:
        return json.load(f)
def dump_json(path:str, file:dict):
    if os.path.exists(path):
        print(f"File already exists at {path}. Skipping dump.")
    with open(path, 'w') as f:
        json.dump(file, f, indent=4)

def get_average_count(count_list:list):
    print(f"Average number of questions for each image: {sum(count_list) / len(count_list)}")

def calculate_stats(input:dict):
    print(f"number of images: {len(input)}")
    original_q_count = []
    new_q_count = []
    for _, val in input.items():
        original_q_count.append(len(val))
        for item in val:
            if 'new_question' in item:
                new_q_count.append(len(item['new_question']))
    print(f"number of original questions in total: {len(new_q_count)}")
    print(f"number of substituted questions in total: {sum(new_q_count)}")
    get_average_count(original_q_count)

def build_question_dic(questions:dict) -> dict:
    '''Take the original data loaded from the json file and output a dictionary grouped by images that contain all the questions (original and substituted ones) and answers. '''
    questions_dict = {}
    for i, (key, val) in enumerate(questions.items()):
        image_id = val['imageId']
        keys = ['questions', 'answer']
        values = [[val['question']], val['answer']]
        each_question_dict = {k: v for k, v in zip(keys, values)}
        if 'new_question' in val:
            each_question_dict['questions'] += val['new_question']
        if 'fullAnswer' in val:
            each_question_dict['fullAnswer'] = val['fullAnswer']
        if 'argument' in val: 
            each_question_dict['argument'] = val['argument']
        if 'hypernym' in val:
            each_question_dict['hypernym'] = val['hypernym']
        each_question_dict['question_type'] = val['types']['detailed']
        each_question_dict['question_id'] = key
        if image_id not in questions_dict:
            questions_dict[image_id] = []
        questions_dict[image_id].append(each_question_dict)
        print(f"\r{i}/{len(questions)} being processed", end='')
    return questions_dict

if __name__ == "__main__":
    data_type = "val"
    # val_question_path = "/projectnb/tin-lab/yuluq/data/subset_combined_stats_data/new_single_nouns.json"
    val_question_path = "/projectnb/tin-lab/yuluq/data/final_gqa/filtered_val/sampled_single_nouns.json"
    train_question_path = "/projectnb/tin-lab/yuluq/data/subset_combined_stats_data/combined_train_single_nouns.json"
    output_path = f"/projectnb/tin-lab/yuluq/data/subset_combined_stats_data/test/0311_{data_type}_question.json"
    if data_type == "train":
        question = load_json(train_question_path)
    elif data_type == "val":
        question = load_json(val_question_path)
    # print(question)
    output = build_question_dic(question)
    # key = output.keys()[0]
    # print(output[key])
    # print(output)
    dump_json(output_path, output)
    calculate_stats(question)




