from multiprocessing import Pool
import re
from utils.util import *
from typing import Dict, List, Any
from functools import partial
from tqdm import tqdm
import spacy

nlp = spacy.load("en_core_web_sm")

def get_proper_imageid(scenegraph, upper_num_obj_bound):
    '''get the list of imageid from that has number of objects ranging from 2-upper_num_obj_bound'''
    proper_imageid = []
    for key, val in scenegraph.items():
        if 2<= len(val['objects']) <= upper_num_obj_bound:
            proper_imageid.append(key)
    return proper_imageid

def lemmatize(words):
    doc = nlp(words)
    return " ".join([token.lemma_ for token in doc])

def check_duplicates(all_names)->bool:
    # deal with example such as palm tree / palm trees
    lemmatized_names = [lemmatize(name) for name in all_names]
    return len(set(lemmatized_names)) != len(lemmatized_names)

def filter_on_scenegraphs(replaced_dic, scenegraph):
    '''only consider scenegraphs where the number of objects is between 2 and 20 and remove those that have duplicate object names'''
    final_dic = {}
    num_not_match_dic = {}
    dup_obj_dic = {}
    none_num = 0
    proper_imgid = get_proper_imageid(scenegraph, 20)
    for i, (key, val) in enumerate(replaced_dic.items()):
        print(f"\r {i}/{len(replaced_dic)}", end="")
        image_id = val.get('imageId')
        item = scenegraph.get(image_id, None)
        if item is not None:
            if image_id in proper_imgid:
                item = scenegraph[image_id]
                all_names = [obj['name'] for obj in item['objects'].values()]
                duplicated_flag = check_duplicates(all_names)
                if not duplicated_flag:
                    final_dic[key] = val
                else:
                    dup_obj_dic[key] = val
            else:
                num_not_match_dic[key] = val  
        else:
            none_num += 1
    return final_dic, num_not_match_dic, dup_obj_dic

def find_non_overlapping_keys(data):
    '''
    Find the keys in the data that do not have overlapping values with any other keys
    Parameter: 
    data: dict[str, list[str]]

    Returns:
    list[str]: keys that do not overlap with any other keys in the data
    
    Example of data: 
        data = {
            "dog": ["canine", "mammal", "animal"],
            "cat": ["feline", "mammal", "animal"],
            "car": ["vehicle"],
            }
    '''
    sets = {k:set(v) for k, v in data.items()}
    return [
        k 
        for k, val in sets.items()
        if all(val.isdisjoint(sets[other_k]) for other_k in sets if other_k != k)
    ]

def get_imgIds(filtered_questions_dict):
    imageIds = set()
    for _, val in filtered_questions_dict.items():
        imageIds.add(val['imageId'])
    return imageIds

def get_scenegraph_proper_args(scenegraphs, hyper_tree, imgIds): 
    '''loop through the filtered imgIds and get the proper args for each image'''
    proper_args = {}
    for id in imgIds:
        name_dict = {}
        # scene-level, check arg names
        for _, entry in scenegraphs[id]['objects'].items():
            arg = entry['name']
            get_base_form(arg)
            if arg in hyper_tree:
                name_dict[arg] = get_hypers(hyper_tree, arg)
        if len(name_dict.keys()) > 1:
            names = find_non_overlapping_keys(name_dict)
            if len(names) != 0:
                proper_args[id] = names
        elif len(name_dict.keys()) == 1:
            proper_args[id] = list(name_dict.keys())
    return proper_args
    
# def get_dict_with_substituions(filtered_single_noun_dic, hyper):

def get_substitutions(filtered_single_noun_dic, hyper_tree):
    '''apply hypernym substitutions to the questions in the filtered_single_noun_dic'''
    pattern = r"(\w+)\s*\((\d+(?:\s*,\s*\d+)*)\)" 
    replaced_dic = {}
    no_hypernym = {}
    # questions = []
    for i, (key, val) in enumerate(filtered_single_noun_dic.items()):
        print(f"\r {i}/{len(filtered_single_noun_dic)}", end="")
        question_str = val['question']
        semantics = val.get('semantic', [])
        # question_dic = {}
        noun, hypers = None, None
        for operation in semantics:
            if operation['operation'] == 'select':
                text = operation['argument']
                match = re.search(pattern, text)
                original_noun = match.group(1)
                noun = get_base_form(original_noun)
            if noun in hyper_tree:
                # hypers = hyper.get(noun, None) #["adult", "person"]
                hypers = get_hypers(hyper_tree, noun)
            else: 
                pass
        if hypers is not None: 
            new_qs = []
            arg_q_forms = []
            for hyp in hypers:
                # updated on May 24
                new_q, new_noun = substitute_in_question(question_str, noun, hyp)
                new_qs.append(new_q)
                arg_q_forms.append(new_noun)
            
            if len(new_qs) > 0:
                val["new_question"] = new_qs
                val["hypernym"] = hypers
                val['argument'] = noun
                # add this information for token analysis, updated on May 24
                # find q_noun for original q
                _, q_noun = substitute_in_question(question_str, noun, original_noun)
                arg_q_forms = [q_noun] + arg_q_forms
                val['arg-q-form'] = arg_q_forms
                replaced_dic[key] = val
            else: 
                no_hypernym[key] = val
    # print(f"filtered single noun dic length: {len(filtered_single_noun_dic)}")
    # print(f"replaced_dic length: {len(replaced_dic)}")
    # print(f"no_hypernym length: {len(no_hypernym)}")
    return replaced_dic, no_hypernym

def has_large_small_long_short_attributes(full_answer)->bool:
    res = False
    # these words are subjective 
    word_list = ["large", "small", "tiny", "huge", "big", "little", "long", "short", "tall"]
    for word in word_list:
        if word in full_answer:
            res = True
            break
    return res

def filter_on_proper_args_and_attributes(last_final_dict, proper_args_dict):
    '''filter the last_final_dict based on the proper_args_dict and ignore the questions that has "large/small" in the fullAnswer'''
    final_dict = {}
    for i, (key, val) in enumerate(last_final_dict.items()):
        print(f"\r {i}/{len(last_final_dict)}", end="")
        proper_args = proper_args_dict.get(val['imageId'], None)
        arg = val['argument']
        full_answer = val['fullAnswer']
        if proper_args is not None:
            if arg in proper_args and not has_large_small_long_short_attributes(full_answer):
                final_dict[key] = val
    return final_dict
  
# Parallel orchestration 
def chunkify_dict(whole_dict: Dict[str, dict], num_chunks: int) -> List[Dict[str, dict]]:
    """
    Split the single noun questions into num_chunks smaller dicts (ceil-balanced)
    """
    items = list(whole_dict.items())
    n, k = len(items), num_chunks
    size = (n+k-1) // k  # Ceiling division to ensure all items are included
    return [dict(items[i:i + size]) for i in range(0, n, size)]

def _process_chunk(chunk: Dict[str, dict], scenegraphs: Dict[str, dict], hyper_tree: Any) -> Dict[str, dict]:
    """
    Process a chunk of single noun questions to filter and substitute hypernyms.
    """
    replaced_dic, _ = get_substitutions(chunk, hyper_tree)
    last_final_dic, _, _ = filter_on_scenegraphs(replaced_dic, scenegraphs)
    new_imgIds = get_imgIds(last_final_dic)
    proper_args = get_scenegraph_proper_args(scenegraphs, hyper_tree, new_imgIds)
    final_dic = filter_on_proper_args_and_attributes(last_final_dic, proper_args)
    return final_dic

def filter_questions(
        single_noun_dict: Dict[str, dict],
        scenegraphs: Dict[str, dict],
        hyper_tree: Any,
        num_workers: int=1,
        num_chunks: int=20
)-> Dict[str, dict]:
    print("Starting to filter questions...")
    if num_workers > 1:
        chunks = chunkify_dict(single_noun_dict, num_chunks)
        worker = partial(_process_chunk, scenegraphs=scenegraphs, hyper_tree=hyper_tree)
        final_dict: Dict[str, dict] = {}
        with Pool(num_workers) as pool:
            for partial_output in tqdm(
                pool.imap(worker, chunks), 
                total=len(chunks),
                desc="Filtering chunks",
                unit="chunk"
            ):
                final_dict.update(partial_output)
        return final_dict
    else:
        # If only one worker, process the entire dictionary without chunking
        return _process_chunk(single_noun_dict, scenegraphs, hyper_tree)
        