#!/usr/bin/env python3
"""
Negative Sampling Script with Relation and Exist Support
Generates 4 negative samples per question, tracking fallbacks.
Supports EXIST, MATERIAL, ATTRIBUTE, and RELATION types.
"""
import random
import re
from pathlib import Path
from collections import defaultdict
import time
from typing import List
import pandas as pd
from utils.util import clean_the_word, substitute_in_question, Tree, read_json
from pipeline.config import QUESTION_TYPE_GROUPS

EXIST_TYPE = set(QUESTION_TYPE_GROUPS["EXIST"])
MATERIAL_TYPES = set(QUESTION_TYPE_GROUPS["MATERIAL"])
ATTRIBUTE_TYPES = set(QUESTION_TYPE_GROUPS["ATTRIBUTE"])
ALL_TYPES = EXIST_TYPE | MATERIAL_TYPES | ATTRIBUTE_TYPES
RELATION_TYPES = set(QUESTION_TYPE_GROUPS["RELATION"])
    
MATERIAL_ATTRIBUTES = {
    "lace", "glass", "metal", "stone", "marble", "wire", "wood",
    "plastic", "leather", "brick", "steel", "copper", "mesh",
    "aluminum", "granite", "concrete", "cloth", "paper", "cardboard",
    "stainless steel", "iron", "wooden"
}

def argument_in_question(question: str, argument: str, before: bool = True) -> bool:
    """
    Check if `argument` appears before (subject) or after (object) the verb phrase.
    """
    m = re.search(r"that is (\w+ing)", question.lower())
    if not m:
        return False
    verb_pos = m.start()
    arg_pos = question.lower().find(argument.lower())
    return arg_pos < verb_pos if before else arg_pos > verb_pos


def identify_potential_args(qtype: str, question: str, argument: str,
                             attr_map: dict,
                             rel_subj_map: dict,
                             rel_obj_map: dict) -> list:
    """
    Return candidate substitutes based on question type.
    For RELATION_TYPES: if an "-ing" verb is detected, use relation maps.
    Otherwise, fall back to MATERIAL or ATTRIBUTE logic.
    """
    tokens = {clean_the_word(w) for w in question.lower().split()}
    text = question.lower()
    # Normalize 'wears' → 'wearing'
    if re.search(r"\bwears\b", text):
        text = re.sub(r"\bwears\b", 'wearing', text)

    # Extract first '-ing' verb
    match = re.search(r"\b(\w+ing)\b", text)
    verb = match.group(1) if match else None

    # RELATION logic
    if qtype in RELATION_TYPES and verb:
        if verb in rel_subj_map and argument_in_question(text, argument, before=True):
            return rel_subj_map.get(verb, [])
        if verb in rel_obj_map and argument_in_question(text, argument, before=False):
            return rel_obj_map.get(verb, [])
        # fall through to ATTR logic

    # MATERIAL logic
    if qtype in MATERIAL_TYPES:
        for tok in tokens:
            if tok in MATERIAL_ATTRIBUTES and tok in attr_map:
                return attr_map[tok]
        return []

    # ATTRIBUTE logic (and fallback for RELATION_TYPES)
    if qtype in ATTRIBUTE_TYPES or qtype in RELATION_TYPES:
        sets = [set(attr_map[k]) for k in tokens if k in attr_map]
        if sets:
            return list(set.intersection(*sets))
        return []

    # Default: no candidates
    return []

def sample_negatives(candidates, negatives, argument, stats):
    pool = set(candidates).intersection(negatives) - {argument}
    if len(pool) < 4:
        try:
            top_category_leaves = Tree[Tree[argument].path()[-2]].leaf_values()
            leaves = set(top_category_leaves)
            # concatenate leaves and pool
            new_pool = pool.union(leaves)
            new_pool = set(new_pool).intersection(negatives)
            if len(new_pool) >=4:
                pool = new_pool
                stats['top_category'] += 1
            else:
                pool = set(negatives)
                stats['random_fallback'] += 1
        except KeyError:
            pool = set(negatives)
            stats['random_fallback'] += 1
    # if len(pool) < 4:
    #     stats['skipped'] += 1
    #     return []
    non_leaf_hyps = Tree[argument].path()[:-1] + Tree[argument].leaf_values()
    final_pool = set(pool) - set(non_leaf_hyps)
    return list(final_pool)

def get_intermediate_nodes(candidates, arg):
    non_leaf_list = []
    non_leaf_hyps = Tree[arg].path()[:-1] + Tree[arg].leaf_values() # the whole chain should be discarded
    print("NON_LEAF HYS", non_leaf_hyps)
    for candidate in candidates:
        try:
            non_leaf_list.extend(Tree[candidate].path()[1:-1])
        except KeyError:
            continue
    final_list = set(non_leaf_list) - set(non_leaf_hyps)
    return list(final_list)

def main(mass_noun, gqa_lemmas, output_dir: Path, NEGATIVE_LEAFS_PATH: Path, NEGATIVE_NON_LEAFS_PATH: Path,
         ATTRIBUTE_OBJECT_MAP_PATH: Path, RELATION_SUBJECT_MAP_PATH: Path,
         RELATION_OBJECT_MAP_PATH: Path, CSV_PATH: Path, seed):
    start_time = time.time()
    random.seed(seed)
    df = pd.read_csv(CSV_PATH, sep=',')
    neg_leaf_map = read_json(Path(NEGATIVE_LEAFS_PATH))
    neg_non_leaf_map = read_json(Path(NEGATIVE_NON_LEAFS_PATH))
    attr_map = read_json(Path(ATTRIBUTE_OBJECT_MAP_PATH))
    rel_subj_map = read_json(Path(RELATION_SUBJECT_MAP_PATH))
    rel_obj_map = read_json(Path(RELATION_OBJECT_MAP_PATH))
    stats = defaultdict(int)
    uncountables = set(get_gqa_lemmas_for_a(gqa_lemmas)) | set(mass_noun)

    for qtype in sorted(ALL_TYPES):
        if qtype not in df['question_type'].unique():
            continue
        subset = df[df['question_type'] == qtype]
        pick_number = 4
        out_rows = []
        for _, row in subset.iterrows():
            substitution_hop, question, arg, original_arg, img_id = row['substitution_hop'], row['question'], row['argument'], row['original_arg'], str(row['image_id'])
            negatives_leaf = set(neg_leaf_map.get(img_id, []))
            negatives_non_leaf = set(neg_non_leaf_map.get(img_id, []))
            if question.startswith("Is there a"):
                negatives_leaf = negatives_leaf - uncountables
                negatives_non_leaf = negatives_non_leaf - uncountables
            # EXIST type: direct random sampling
            if qtype in EXIST_TYPE:
                non_leaf_hyps = set(Tree[original_arg].path()[:-1] + Tree[original_arg].leaf_values())
                if substitution_hop == 0:
                    if len(negatives_leaf) < 4:
                        stats['skipped'] += 1
                        continue
                    negatives_leaf = list(negatives_leaf - non_leaf_hyps)
                    picks = random.sample(negatives_leaf, pick_number)
                else:
                    if len(negatives_non_leaf) < 4 and len(negatives_leaf) < 4:
                        stats['skipped'] += 1
                        continue
                    # concatenate leaf and non-leaf negatives
                    negatives = list(negatives_leaf | negatives_non_leaf - non_leaf_hyps)
                    picks = random.sample(negatives, pick_number)
            elif qtype in MATERIAL_TYPES:
                pick_number = 4
                cands = identify_potential_args(
                    qtype, question, arg,
                    attr_map, rel_subj_map, rel_obj_map
                )

                # get the intermediate nodes for all the candidates
                if substitution_hop == 0:
                    print("Negative leafs: ", negatives_leaf)
                    print("Candidates: ", cands)
                    print("ARGUMENT: ", arg)    
                    picks = sample_negatives(cands, negatives_leaf, original_arg, stats)
                    if len(picks) < pick_number:
                        stats['skipped'] += 1
                        continue
                    picks = random.sample(picks, pick_number)
                    print(f"DEBUGGING PURPOSE for LEAFs: FOR WIRE! PICKES INCLUDE: {picks}")
                else:
                    picks = sample_negatives(cands, negatives_leaf, original_arg, stats)
                    non_leaf_picks = get_intermediate_nodes(picks, original_arg)
                    final_pics = list(set(picks + non_leaf_picks))
                    if len(final_pics) < pick_number:
                            stats['skipped'] += 1
                            continue
                    else:
                        picks = random.sample(final_pics, pick_number)
                    print(f"DEBUGGING PURPOSE for NON-leafs: FOR WIRE! FINAL PICKS INCLUDE: {picks}")
            else:
                cands = identify_potential_args(
                    qtype, question, arg,
                    attr_map, rel_subj_map, rel_obj_map
                )

                # get the intermediate nodes for all the candidates
                if substitution_hop == 0:
                    picks = sample_negatives(cands, negatives_leaf, original_arg, stats)
                    if len(picks) < pick_number:
                        stats['skipped'] += 1
                        continue
                    picks = random.sample(picks, pick_number)
                else:
                    picks = sample_negatives(cands, negatives_leaf, original_arg, stats)
                    non_leaf_picks = get_intermediate_nodes(picks, original_arg)
                    
                    if arg in picks:
                        print("BUGGY, arg in leafs")
                    if arg in non_leaf_picks:
                        print("BUGGY, arg in non-leafs")
                    final_pics = list(set(picks + non_leaf_picks))
                    if arg in final_pics:
                        print("TRUE, bugggggy!")
                        final_pics.remove(arg)
                        
                    if len(final_pics) < pick_number:
                        stats['skipped'] += 1
                        continue
                    else:
                        picks = random.sample(final_pics, pick_number)
            stats['processed'] += 1
            for neg in picks:
                if neg == arg:
                    print(picks)
                    stats['bad'] += 1
                    print(neg, arg)
                    print("THIS IS THE BUGGy qtype", qtype)

                if substitution_hop > 0:
                    question = row['original_question']
                    arg = row['original_arg']
                new_q, new_noun = substitute_in_question(question, arg, neg)
                if new_q == question:
                    print(f"Substitution failed: {question} -> {new_q}")
                out = row.copy()
                out['question'] = new_q
                out['argument'] = neg
                out['arg-scene-form'] = original_arg
                out['arg-q-form'] = new_noun
                if substitution_hop == 0:
                    out['substitution_hop'] = -100
                else:
                    out['substitution_hop'] = -substitution_hop
                out['ground_truth'] = 'no'
                out.drop(labels=['ground_truth_long'], errors='ignore', inplace=True)
                out_rows.append(out)

        pd.DataFrame(out_rows).to_csv(
            Path(output_dir)/f"{qtype}_negative_samples.csv",
            sep=',', index=False
        )
        print(f"{qtype}: {len(out_rows)} generated ({stats['skipped']} skipped)")

    print(f"Top-category fallback count: {stats['top_category']}")
    print(f"Random fallback count: {stats['random_fallback']}")
    print(f"Skipped count: {stats['skipped']}")
    print(f"Total processed: {stats['processed']}" )
    print(f"Bad substitutions: {stats['bad']}")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("Done!")

def get_gqa_lemmas_for_a(gqa_lemmas)-> List:
    uncountables = (
    gqa_lemmas[
        gqa_lemmas['lemma'] == gqa_lemmas['article']
        ]['lemma']
        .tolist()
    )
    return uncountables

def negative_sampling(mass_noun, gqa_lemmas, output_dir: Path, negative_leafs_path: Path, negative_non_leafs_path: Path,
                      attribute_object_map_path: Path, relation_subject_map_path: Path,
                      relation_object_map_path: Path, csv_path: Path, seed: int = 42):
    """
    Main function to run the negative sampling pipeline.
    """
    print("Starting negative sampling...")
        # --- Constants ---
    # if output_dir does not exist, create it
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    main(mass_noun, gqa_lemmas, output_dir, negative_leafs_path, negative_non_leafs_path,
         attribute_object_map_path, relation_subject_map_path,
         relation_object_map_path, csv_path, seed)
    print("Negative sampling completed.")
    print("merge negative samples with original questions")
    final_df = merge_negative_samples(output_dir, csv_path)
    return final_df

def merge_negative_samples(negative_dir: Path, csv_path):
    # 1) load negative samples and merge them together
    negative_paths = negative_dir.glob("*.csv")
    dfs = [
        pd.read_csv(p, sep=',')   # or sep=',' depending on what you used
        for p in negative_paths
    ]
    merged = pd.concat(dfs, ignore_index=True)
    print("number of rows in the merged files: ", len(merged))
    # 2) read the original question csv file
    original_questions = pd.read_csv(csv_path, sep=",")
    # get the types
    original_questions = original_questions[original_questions['question_type'].isin(ALL_TYPES)]
    # 3) concatenate them
    final_df = pd.concat([original_questions, merged], ignore_index=True)
    print(f"Merged {len(dfs)} files into original_questions")
    print(f"Final dataframe has {len(final_df)} rows.")
    return final_df