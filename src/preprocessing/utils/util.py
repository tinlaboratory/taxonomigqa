
import spacy
import inflect
import string
import re

import json
from semantic_memory import taxonomy
from collections import defaultdict
from pipeline.config import HYPERNYM_PATH
import pandas as pd

nlp = spacy.load("en_core_web_sm")  # Load spaCy English model
inflector = inflect.engine()

def get_base_form(word):
    doc = nlp(word)
    return doc[0].lemma_
def clean_the_word(word):
    return word.strip(string.punctuation).lower()

def detect_target_word(question, arg):
    words = question.split(' ')
    target_word = None

    for word in words:
        clean_word = clean_the_word(word)
        if arg in clean_word: #also include cases like 't-shirt'
            target_word = clean_word
            break
    # second pass, to save checking with spacy
    if target_word == None:
        for word in words:
            clean_word = clean_the_word(word)
            base_form = get_base_form(clean_word)
            if arg in base_form: # for men, women
                target_word = clean_word
    return target_word

def is_plural(word):
    return inflector.singular_noun(word) is not False

def substitute_in_question(question, noun, substituted_word):
    # 1) detect the exact form you matched (handles "'s" too)
    target = detect_target_word(question, noun)
    
    if target is None:
        print("Target word not found in question.", noun, question, substituted_word)
        print("Failing!!!!")
        return question

    # strip possessive "'s" so we treat "dog's" like "dog"
    possessive = False
    if target.lower().endswith("'s"):
        target = target[:-2]
        possessive = True

    # 2) decide plurality & get the right form of substituted_word
    if is_plural(target) and target.lower() != noun.lower():
        new_noun = inflector.plural(substituted_word)
    else:
        if question.startswith("Are there"): # deal with edge case, such as Are there any sheep in the image? 
            new_noun = inflector.plural(substituted_word)
        else:
            new_noun = substituted_word
    
    # 3) fix any preceding “a” or “an”
    # only singular nouns ever get “a/an”
    if not is_plural(target):
        art_pat = re.compile(
            rf"\b(a|an)\s+{re.escape(target)}\b",
            flags=re.IGNORECASE
        )
        def art_repl(m):
            # inflector.a gives e.g. "an elephant" or "a cat"
            rep = inflector.a(substituted_word)
            # preserve capitalization of the original article
            if m.group(1)[0].isupper():
                rep = rep.capitalize()
            return rep

        question = art_pat.sub(art_repl, question)

    # 4) replace any leftover bare occurrences of the noun
    noun_pat = re.compile(rf"\b{re.escape(target)}\b", flags=re.IGNORECASE)
    def noun_repl(m):
        # preserve capitalization of the original word
        w = new_noun
        if m.group(0)[0].isupper():
            w = w.capitalize()
        return w

    new_q = noun_pat.sub(noun_repl, question)

    # 5) if it was possessive, re-add "'s"
    if possessive:
        # e.g. dog’s → substitute to elephant’s
        new_q = re.sub(
            rf"{re.escape(new_noun)}'s\b",
            new_noun + "'s",
            new_q,
            flags=re.IGNORECASE
        )
    return new_q, new_noun


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
noun_hypernyms = read_json(HYPERNYM_PATH)
hypernym_paths = defaultdict(set)

for noun, hypernyms in noun_hypernyms.items():
    hypernym_paths[noun].add(tuple(hypernyms))
    # each hypernym is the child of the next one
    for i in range(len(hypernyms) - 1):
        hypernym_paths[hypernyms[i]].add(tuple(hypernyms[i + 1:]))

# store only the longest paths
longest_paths = {}
for noun, paths in hypernym_paths.items():
    longest_paths[noun] = max(paths, key=len)

# now store the unique hypernym pairs
hypernym_pairs = {}
for noun, path in longest_paths.items():
    hypernym_pairs[noun] = path[0]

Tree = taxonomy.Nodeset(taxonomy.Node)
root = Tree['ROOT']

# # populate the tree

for concept, path in hypernym_pairs.items():
    node = Tree[concept]
    parent = Tree[path]
    node.add_parent(parent)
    parent.add_child(node)

# make sure root is added as a parent to all top level nodes
for value, node in Tree.items():
    if value == "ROOT":
        continue
    elif node.parent is None:
        node.add_parent(root)
        root.add_child(node)

# Tree.default_factory = None
Tree.default_factory = None

def get_hypers(hyper_tree, noun):
    return hyper_tree[noun].path()[1:-1]

def get_proper_args_for_negative_sampling(scenegraph, hyper_tree, text_prompts_df):
    """
    Prepare the arguments for negative sampling based on the scenegraph and hyper_tree.
    
    Args:
        scenegraph (dict): The scenegraph data.
        hyper_tree (taxonomy.Nodeset): The hypernym tree.
        text_prompts_df (pd.DataFrame): DataFrame containing text prompts.
        
    Returns:
        tuple: Two dictionaries, one includes leaf nodes and one includes non-leaf nodes. Key is the image_id, value is a list of object candidates for negative sampling.
    """
    # get all the object names in the scenegraph
    # get all the hypers for each arg in the scenegraph
    # get the imageIds from text_prompts_df
    image_ids = text_prompts_df['image_id'].unique()
    all_args = text_prompts_df['original_arg'].unique()
    all_args_hyp = text_prompts_df[
        (text_prompts_df['substitution_hop'] != 0) 
    ]['argument'].unique()

    # get the object names from the scenegraph
    neg_leaf_candidates = {}
    neg_non_leaf_candidates = {}
    for id in image_ids:
        neg_leaf_candidates[id] = []
        neg_non_leaf_candidates[id] = []

        item = scenegraph[id]
        # base-form please
        all_names = [get_base_form(obj['name']) for obj in item['objects'].values()]
        # get the hypers for each name
        hyper_list = []
        for name in all_names:
            if name in hyper_tree:
                # add the hypers to the hyper_list
                hyper_list.extend(get_hypers(hyper_tree, name))
        leaf_candidates = set(all_args) - set(all_names) - set(hyper_list) # to be more precise
        non_leaf_candidates = set(all_args_hyp) - set(hyper_list) - set(all_names)
        neg_leaf_candidates[id] = list(leaf_candidates)
        neg_non_leaf_candidates[id] = list(non_leaf_candidates)
    return neg_leaf_candidates, neg_non_leaf_candidates

def format_scene_description(
    scene_texts):
    for key, val in scene_texts.items():
        descriptions = " ".join(val)
        scene_texts[key] = descriptions
    return scene_texts

        






