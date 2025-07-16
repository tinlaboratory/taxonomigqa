# imageId: text description.
import json
import random
from random import sample
import pandas as pd
from typing import Dict, List

import spacy

nlp = spacy.load("en_core_web_sm")

def load_json(file_path: str) -> dict:
    """load json file"""
    with open(file_path) as f:
        return json.load(f)

def sample_relations(relations: list) -> dict:
    """Take the original relation list, group the repeated relations, sample 1 randomly and return a new list of relations that contains only 1 relation for every object"""
    new_dict = {}
    for relation in relations:
        obj_id = relation["object"]
        if obj_id not in new_dict.keys():
            new_dict[obj_id] = []
            new_dict[obj_id].append(relation["name"])
        else:
            new_dict[obj_id].append(relation["name"])
    for obj_id, relation in new_dict.items():
        if len(relation) > 1:
            sample_data = sample(relation, 1)
            new_dict[obj_id] = sample_data
    return new_dict

def order_objects_from_large_to_small(objs: dict) -> dict:
    """loop over the objects, put the ids in descending order based on the area of the bounding boxes"""

    ordered_objs = dict(
        sorted(objs.items(), key=lambda item: item[1]["w"] * item[1]["h"], reverse=True)
    )
    return ordered_objs


def order_objects_based_on_number_of_relations(objs: dict) -> dict:
    """loop over the objects, sorted the objects based on nubmer of relations they have in ascending order"""
    ordered_objs = dict(
        sorted(objs.items(), key=lambda item: len(item[1]["relations"]))
    )
    return ordered_objs

def add_determiner_for_countable_nouns(
    obj_tag: str, name_with_attribute: str
) -> str:
    if obj_tag not in ["NNS", "VBZ", "NNPS"]:
                        if name_with_attribute[0].lower() in ["a", "e", "i", "o", "u"]:
                            name_with_attribute = "an " + name_with_attribute
                        else:
                            name_with_attribute = "a " + name_with_attribute
    return name_with_attribute

def preprocess_scenegraph(d: dict, img_ids: list, scene_descs: list, mass_nouns, gqa_lemmas):
    all_names = []
    for i, img_id in enumerate(img_ids):
        print(f"\r {i}/{len(img_ids)} processed", end="")
        val = d[img_id]
        object_d = val["objects"]
        object_bbx_order = order_objects_from_large_to_small(val["objects"])

        descs = []
        intro_str = f"There are {len(object_d)} objects in the scene, including "
        names = [val["name"] for val in object_bbx_order.values()]

        attributes = [" ".join(dict.fromkeys(val["attributes"])) for val in object_bbx_order.values()] # remove duplicates and keep attribute order
        for i, name in enumerate(names):
            name_with_attribute = (
                f"{attributes[i] + ' ' if attributes[i]!='' else ''}" + name
            )
            obj_tag = nlp(name)[-1].tag_ # deal with phrases such as white trees, palm trees

            ####################### small block to focus on ###########################
            # check if name is in gqa_lemmas
            if name in gqa_lemmas:
                if gqa_lemmas[name] != name:
                    name_with_attribute = add_determiner_for_countable_nouns(
                        obj_tag, name_with_attribute
                    )
            else:
                if name not in mass_nouns:
                    name_with_attribute = add_determiner_for_countable_nouns(
                        obj_tag, name_with_attribute
                    )

            if i == len(names) - 1:
                intro_str += f"and {name_with_attribute}."
            else:
                intro_str += f"{name_with_attribute}, "
            ################### small block to focus on ###########################
        descs.append(intro_str)

        for obj_key, obj_val in object_bbx_order.items():
            obj_name = obj_val["name"]
            obj_attributes = ", ".join(obj_val["attributes"])
            obj_tag = nlp(obj_name)[-1].tag_
            obj_name_with_attributes = ""
            if obj_attributes:
                obj_name_with_attributes = obj_attributes + " " + obj_name
            else:
                obj_name_with_attributes = obj_name

            if obj_tag in ["NNS", "VBZ", "NNPS"]:
                obj_number = "plural"
            else:
                obj_number = "singular"

            relations = sample_relations(obj_val["relations"])
            for obj_id, rel in relations.items():
                rel_obj_name = object_d[obj_id]["name"]
                rel_label = rel[0]
                string_desc = None
                if rel_label == "of":
                    if obj_number == "singular":
                        string_desc = f"The {obj_name} belongs to the {rel_obj_name}."
                    else:
                        string_desc = f"The {obj_name} belong to the {rel_obj_name}."
                else:
                    if obj_number == "singular":
                        string_desc = (
                            f"The {obj_name} is {rel_label} the {rel_obj_name}."
                        )
                    else:
                        string_desc = (
                            f"The {obj_name} are {rel_label} the {rel_obj_name}."
                        )
                descs.append(string_desc)
        scene_descs[img_id] = descs

def generate_scene_description(
        scenegraph: Dict,
        image_ids: List, 
        mass_nouns: List,
        gqa_lemmas: pd.DataFrame,
        random_seed: int=42
):
    """Main function to generate scene descriptions from the scene graph data."""
    # want a mapping that has lemma as the key and the value from column article
    gqa_lemmas = gqa_lemmas.set_index("lemma")["article"].to_dict()
    scene_descs = {}
    # add new logic
    random.seed(random_seed)
    preprocess_scenegraph(scenegraph, image_ids, scene_descs, mass_nouns, gqa_lemmas)
    return scene_descs
