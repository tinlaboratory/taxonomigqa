import json
from pathlib import Path
import re
import json
import os
import glob

def save_to_json(data_dict, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through each dictionary and filename pair
    for filename, data in data_dict.items():
        file_path = os.path.join(output_folder, filename)
        with open(file_path, 'w') as file:
            json.dump(data, file)
        print(f'Saved data to {file_path}')

def main():
    input_folder = Path("/projectnb/tin-lab/yuluq/data/gqa/data/train_all_questions/")
    for i, file_path in enumerate(glob.glob(os.path.join(input_folder, '*.json'))):
        with open(file_path, 'r') as file:
            question = json.load(file)
        # first filter
        filtered_single_noun_dic = {}
        filtered_more_nouns_dic = {}
        multiple_mention_dic = {}
        half_or_no_mention_dic = {}
        global_dic = {}
        noun_not_in_q = {}
        max_noun_num = 0

        pattern = r"(\w+)\s*\((\d+(?:\s*,\s*\d+)*)\)" # should be a correct one
        for key, val in question.items():
            
            question = val['question']
            question_type = val['types']['semantic']
            nouns = []
            if question_type != 'global':
                overall_flag = True
                multiple_mention_flag = False
                half_or_no_mention_flag = False
                noun_not_in_q_flag = False
                semantics = val.get('semantic', [])
                for operation in semantics:
                    if operation['operation'] == 'select':
                        text = operation['argument']
                        match = re.search(pattern, text)
                        if match: 
                            noun = match.group(1)
                            if noun not in question: 
                                overall_flag = False
                                noun_not_in_q_flag = True
                            else:
                                numbers = match.group(2)
                                number_list = [n.strip() for n in numbers.split(',')]
                                if len(number_list) > 1:
                                    # as long as there is a multiple mention, add this case
                                    multiple_mention_flag = True
                                    overall_flag = False
                                nouns.append(noun)
                        else: 
                            overall_flag = False
                            half_or_no_mention_flag = True
                if len(nouns) > max_noun_num:
                    max_noun_num = len(nouns)
                if overall_flag:
                    if len(nouns) > 1:
                        filtered_more_nouns_dic[key] = val
                    else:
                        filtered_single_noun_dic[key] = val
                elif multiple_mention_flag:
                    multiple_mention_dic[key] = val
                elif half_or_no_mention_flag:
                    half_or_no_mention_dic[key] = val  
                elif noun_not_in_q_flag:
                    noun_not_in_q[key] = val
            else:
                global_dic[key] = val       
                
        output_folder = '/projectnb/tin-lab/yuluq/data/fgqa/train/'
        data_to_save = {
            f'train_single_nouns_{i}.json': filtered_single_noun_dic,
            # 'more_nouns.json': filtered_more_nouns_dic,
            # 'multiple_references_dic.json': multiple_mention_dic,
            # 'half_mention.json': half_or_no_mention_dic,
            # 'global_questions.json': global_dic,
            # 'noun_not_in_q.json': noun_not_in_q,
        }
        save_to_json(data_to_save, output_folder)

if __name__ == "__main__":
    main()