import json
import re
from pathlib import Path
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
    
# load the scenegraph
def load_scenegraphs(split):
    path = f"../data/sceneGraphs/{split}_sceneGraphs.json"
    with open(path) as f:
        d = json.load(f)
    return d

def main():
    with open('/projectnb/tin-lab/yuluq/multimodal-representations/data/gqa_entities/noun-hypernyms.json') as f:
        hyper = json.load(f)
    input_folder = Path("/projectnb/tin-lab/yuluq/data/fgqa/train") 
    
    for i, file_path in enumerate(glob.glob(os.path.join(input_folder, "*.json"))):
        with open(file_path, 'r') as file:
            filtered_single_noun_dic = json.load(file)
            
    # --------------------------- val -----------------------------------
    # with open("/projectnb/tin-lab/yuluq/data/fgqa/val_balanced/single_nouns.json", 'r') as f:
    #     filtered_single_noun_dic = json.load(f)
        output_folder = Path('/projectnb/tin-lab/yuluq/data/rgqa/train')
        pattern = r"(\w+)\s*\((\d+(?:\s*,\s*\d+)*)\)" # should be a correct one
        replaced_dic = {}
        no_hypernym = {}
        for key, val in filtered_single_noun_dic.items():
            question_str = val['question']
            semantics = val.get('semantic', [])
            # question_dic = {}
            for operation in semantics:
                if operation['operation'] == 'select':
                    text = operation['argument']
                    match = re.search(pattern, text)
                    noun = match.group(1)
            hypers = hyper.get(noun, None) #["adult", "person"]
            if hypers is not None: 
                new_qs = []
                for hyp in hypers:
                    new_q = question_str.replace(noun, hyp)
                    new_qs.append(new_q)
                # question_dic['original'] = question_str
                # question_dic['new'] = new_qs
                # question_dic['arg'] = noun
                # question_dic['hypernym'] = hypers
                # question_dic['answer'] = val['answer']
                # question_dic['fullAnswer'] = val['fullAnswer']
                # question_dic['imageId'] = val['imageId']
            # if len(question_dic) > 0:
            if len(new_qs) > 0:
                val["new_qeustion"] = new_qs
                val["hypernym"] = hypers
                replaced_dic[key] = val
            else: 
                no_hypernym[key] = val
                
        val_scenegraphs = load_scenegraphs("train")
        final_dic = {}
        num_not_match_dic = {}
        dup_obj_dic = {}
        none_num = 0
        for key, val in replaced_dic.items():
            image_id = val.get('imageId')
            item = val_scenegraphs.get(image_id, None)
            if item is not None:
                # if len(item['objects']) >=2 and len(item['objects']) <= 20:
                if 2 <= len(item['objects']) <=20:
                    all_names = [obj['name'] for obj in item['objects'].values()]
                    duplicated_flag = len(set(all_names))!=len(all_names)
                    # name_set = set()
                    # duplicated_flag = False
                    # for key_1, objs in item['objects'].items():
                    #     if objs['name'] not in name_set:
                    #         name_set.add(objs['name'])
                    #     else:
                    #         duplicated_flag = True
                    if not duplicated_flag:
                        final_dic[key] = val
                    else:
                        dup_obj_dic[key] = val
                    # final_dic[key] = val
                else:
                    num_not_match_dic[key] = val  
            else:
                none_num += 1
        data_to_save = {f"train_single_nouns_{i}.json": final_dic}
        # with open(Path(output_folder, 'single_nouns_balanced.json'), 'w') as f:
        #     json.dump(final_dic, f)
        save_to_json(data_to_save, output_folder)
if __name__ == "__main__":
    main()