import re
from typing import Dict
from tqdm import tqdm

PATTERN = r"(\w+)\s*\((\d+(?:\s*,\s*\d+)*)\)" 

def extract_single_nouns(question) -> Dict[str, Dict]:
    print("Extracting single nouns from questions...")
    filtered_single_noun_dic = {}
    filtered_more_nouns_dic = {}
    multiple_mention_dic = {}
    half_or_no_mention_dic = {}
    global_dic = {}
    noun_not_in_q = {}

    for key, val in tqdm(question.items(), desc="Processing questions"):
        question = val['question']
        question_type = val['types']['semantic']
        nouns = []
        if question_type != 'global':
            single_or_more_noun_flag = True
            multiple_mention_flag = False
            half_or_no_mention_flag = False
            noun_not_in_q_flag = False
            semantics = val.get('semantic', [])
            for operation in semantics:
                if operation['operation'] == 'select':
                    text = operation['argument']
                    match = re.search(PATTERN, text)
                    if match: 
                        noun = match.group(1)
                        if noun not in question: 
                            single_or_more_noun_flag = False
                            noun_not_in_q_flag = True
                        else:
                            numbers = match.group(2)
                            number_list = [n.strip() for n in numbers.split(',')]
                            if len(number_list) > 1:
                                # as long as there is a multiple mention, add this case
                                multiple_mention_flag = True
                                single_or_more_noun_flag = False
                            nouns.append(noun)
                    else: 
                        single_or_more_noun_flag = False
                        half_or_no_mention_flag = True

            if single_or_more_noun_flag:
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
    
    return {
        'single_noun': filtered_single_noun_dic,
        'more_nouns': filtered_more_nouns_dic,
        'multiple_mention': multiple_mention_dic,
        'half_or_no_mention': half_or_no_mention_dic,
        'global': global_dic,
        'noun_not_in_q': noun_not_in_q,
    }
