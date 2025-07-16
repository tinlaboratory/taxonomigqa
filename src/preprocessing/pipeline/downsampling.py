from tqdm import tqdm
from collections import OrderedDict
from typing import Dict
import random

def group_single_nouns_by_imgid(single_nouns: dict) -> OrderedDict:
    grouped = OrderedDict()
    for key in sorted(single_nouns):  # sort question ids
        item = single_nouns[key]
        imageid = item['imageId']
        dict_entry = {
            'question': item['question'],
            'imageId': imageid,
            'answer': item['answer'],
            'types': item['types']['detailed'],
            'fullAnswer': item['fullAnswer'],
            'new_question': item['new_question'],
            'hypernym': item['hypernym'],
            'argument': item['argument'],
            'question_id': key
        }
        if imageid not in grouped:
            grouped[imageid] = []
        grouped[imageid].append(dict_entry)
    return grouped

def get_full_levels_and_rest_question_ids(data_per_img: list):
    full_level_ids = []
    rest_ids = []
    for q in data_per_img:
        if len(q['new_question']) >= 4:
            full_level_ids.append(q['question_id'])
        else:
            rest_ids.append(q['question_id'])
    if len(full_level_ids) > 40:
        full_level_ids = sorted(random.sample(full_level_ids, 40))  # sort for stability
    return full_level_ids, rest_ids

def get_question_type_probability_and_ids(data_per_img: list, q_ids_to_consider: list):
    question_type_ids = OrderedDict()
    for q in data_per_img:
        qid = q['question_id']
        if qid in q_ids_to_consider:
            qtype = q['types']
            if qtype not in question_type_ids:
                question_type_ids[qtype] = []
            question_type_ids[qtype].append(qid)

    total = sum(len(v) for v in question_type_ids.values())
    sampling_probs = {k: len(v) / total for k, v in question_type_ids.items()}
    return question_type_ids, sampling_probs

def sample_from_question_types(data_per_img: list, q_ids_to_consider: list, k: int):
    qtype_ids, probs = get_question_type_probability_and_ids(data_per_img, q_ids_to_consider)
    qtypes = list(qtype_ids.keys())
    weights = [probs[qtype] for qtype in qtypes]
    sampled_items = []
    seen = set()

    if len(q_ids_to_consider) < k:
        k = len(q_ids_to_consider)

    while len(sampled_items) < k:
        chosen_type = random.choices(qtypes, weights=weights, k=1)[0]
        candidates = sorted(qtype_ids[chosen_type])  # sort to make choice stable
        candidate = random.choice(candidates)
        if candidate not in seen:
            sampled_items.append(candidate)
            seen.add(candidate)

    return sorted(sampled_items)  # sort final list for determinism

def downsample_questions(
    questions: Dict[str, dict], 
    max_sample_size: int=40,
    random_seed: int=42
)->OrderedDict[str, dict]:
    random.seed(random_seed)
    original_data = questions
    grouped_data = group_single_nouns_by_imgid(original_data)

    dict_for_sampling = OrderedDict((imgid, []) for imgid in sorted(grouped_data))

    for imgid in tqdm(dict_for_sampling):
        full_ids, rest_ids = get_full_levels_and_rest_question_ids(grouped_data[imgid])
        dict_for_sampling[imgid] = full_ids
        remaining = max_sample_size - len(full_ids)
        if remaining > 0 and rest_ids:
            sampled = sample_from_question_types(grouped_data[imgid], rest_ids, remaining)
            dict_for_sampling[imgid].extend(sampled)

    # Filter final questions
    filtered_data = OrderedDict()
    for imgid in dict_for_sampling:
        for qid in sorted(dict_for_sampling[imgid]):  # sort qids for determinism
            filtered_data[qid] = original_data[qid]

    return filtered_data
