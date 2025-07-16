# this file is to build a dataframe that stores all the necessary information for prompting and evaluation
import json
import pandas as pd
import re
from typing import Dict, List

def replace_text(text, _pattern):
    # a) remove those exact phrases
    s = _pattern.sub("", text)
    # b) collapse any run of spaces into one
    s = re.sub(r"\s+", " ", s)
    # c) remove space before punctuation
    s = re.sub(r"\s+([?!.])", r"\1", s)
    return s.strip()

def build_question_dic(questions: dict) -> dict:
    """Take the original data loaded from the json file and output a dictionary grouped by images that contain all the questions (original and substituted ones) and answers."""
    questions_dict = {}
    for i, (key, val) in enumerate(questions.items()):
        image_id = val["imageId"]
        keys = ["questions", "answer"]
        values = [[val["question"]], val["answer"]]
        each_question_dict = {k: v for k, v in zip(keys, values)}
        if "new_question" in val:
            each_question_dict["questions"] += val["new_question"]
        if "fullAnswer" in val:
            each_question_dict["fullAnswer"] = val["fullAnswer"]
        if "argument" in val:
            each_question_dict["argument"] = val["argument"]
        if "hypernym" in val:
            each_question_dict["hypernym"] = val["hypernym"]
        if "arg-q-form" in val:
            each_question_dict["arg-q-form"] = val["arg-q-form"]
        each_question_dict["question_type"] = val["types"]["detailed"]
        each_question_dict["question_id"] = key
        if image_id not in questions_dict:
            questions_dict[image_id] = []
        questions_dict[image_id].append(each_question_dict)
        print(f"\r{i}/{len(questions)} being processed", end="")
    return questions_dict

def prompt_builder(
        val_qs: dict, 
        scene_descriptions: Dict[str, List[str]],
        image_as_modality, 
        ) -> pd.DataFrame:
     # load scene descriptions as list of strings
    val_qs = build_question_dic(val_qs)
    # 1) enumerate exactly the three-word phrases you want to drop:
    prefixes = ["in the", "in this", "of the", "of this"]
    nouns    = ["image", "photo", "photograph", "picture", "scene"]
    phrases  = [f"{p} {n}" for p in prefixes for n in nouns]

    # 2) compile a regex that only matches those phrases when they stand alone:
    _pattern = re.compile(
        r'(?:(?<=\s)|^)'               # must be preceded by whitespace or start-of-string
        r'(?:' + '|'.join(map(re.escape, phrases)) + r')'  # one of our allowed three-word phrases
        r'(?=(?:\s|[?!.]|$))'           # must be followed by whitespace, punctuation, or end-of-string
        , flags=re.IGNORECASE
    )

    rows = []
    question_id = None
    for i, (key, val) in enumerate(val_qs.items()):
        for entry in val:
            image_id = key
            question = ""
            scene_description = scene_descriptions[key]
            args = [entry["argument"]] + entry["hypernym"]
            questions = entry["questions"]
            ground_truth = entry["answer"]
            ground_truth_long = entry["fullAnswer"]
            argument = entry["argument"]
            original_arg = entry["argument"]
            noun_in_qs = entry['arg-q-form']
            if not image_as_modality:
                original_question = replace_text(questions[0], _pattern)
            else:
                original_question = questions[0]
            for k, (arg, question) in enumerate(zip(args, questions)):
                question_id = entry["question_id"]
                noun_in_q = noun_in_qs[k]
                if not image_as_modality:
                    question = replace_text(question, _pattern)
                question_type = entry["question_type"]
                substitution_hop = k
                dict_to_be_added = {
                    "question_id": question_id, 
                    "image_id": image_id, 
                    "question" : question, 
                    "original_question": original_question,
                    "question_type": question_type,
                    "substitution_hop": substitution_hop,
                    "argument": arg,
                    "original_arg": original_arg,
                    "arg-scene-form": original_arg,
                    "arg-q-form": noun_in_q,
                    "scene_description": scene_description,
                    "ground_truth": ground_truth,
                    "ground_truth_long": ground_truth_long,
                }
                rows.append(dict_to_be_added)

    df = pd.DataFrame(rows)
    df = df[df['ground_truth'].str.lower().isin(['yes', 'no'])]
    return df