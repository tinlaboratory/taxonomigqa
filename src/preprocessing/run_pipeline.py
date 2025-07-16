from pathlib import Path
from utils.util import read_json, get_proper_args_for_negative_sampling, format_scene_description
from utils.tree_util import get_tree
from pipeline.extract_single_nouns import extract_single_nouns
from pipeline.filter_questions import filter_questions
from pipeline.downsampling import downsample_questions
from pipeline.scene_processor import generate_scene_description
from pipeline.prompt_builder import prompt_builder
from pipeline.negative_sampling import negative_sampling
import json
import pandas as pd
import time
from pipeline.config import (
    VAL_QUESTIONS,
    VAL_SCENEGRAPHS,
    MASS_NOUNS_JSON,
    GQA_LEMMAS_CSV,
    IMAGE_PROMPTS_TSV,
    TEXT_PROMPTS_TSV,
    NEG_LEAF_JSON,
    NEG_NON_LEAF_JSON,
    NEGATIVE_SAMPLES_DIR,
    ATTRIBUTE_OBJECT_MAP_PATH,
    RELATION_SUBJECT_MAP_PATH,
    RELATION_OBJECT_MAP_PATH,
    HYPERNYM_PATH,
    # Intermediate paths
    FILTERED_QUESTIONS_JSON,
    DOWNSAMPLING_QUESTIONS_JSON, 
    SCENE_DESCRIPTIONS_JSON, 
    # Final data file path
    FINAL_DATA_FILE_PATH_IMAGE,
    FINAL_DATA_FILE_PATH_TEXT, 
    # Question types
    QUESTION_TYPE_GROUPS
)

EXIST_TYPE = set(QUESTION_TYPE_GROUPS["EXIST"])
MATERIAL_TYPES = set(QUESTION_TYPE_GROUPS["MATERIAL"])
ATTRIBUTE_TYPES = set(QUESTION_TYPE_GROUPS["ATTRIBUTE"])
ALL_TYPES = EXIST_TYPE | MATERIAL_TYPES | ATTRIBUTE_TYPES

def main():
    start_time = time.time()
    split = "val"
    seed = 42
    gqa_raw_qs = read_json(VAL_QUESTIONS)
    hyper_tree = get_tree(HYPERNYM_PATH)
    scenegraph = read_json(VAL_SCENEGRAPHS)
    mass_nouns = read_json(MASS_NOUNS_JSON)
    gqa_lemmas = pd.read_csv(GQA_LEMMAS_CSV)
 
    # STEP 1: remove global type questions and extract single noun questions
    single_noun = extract_single_nouns(gqa_raw_qs)['single_noun']
    print(f"Extracted {len(single_noun)} single noun questions.")
    # STEP 2: filter questions 

    # check if the file exists
    if not Path(FILTERED_QUESTIONS_JSON).exists():
        filtered = filter_questions(
            single_noun_dict=single_noun,
            scenegraphs=scenegraph,
            hyper_tree=hyper_tree,
            num_workers=8,
            num_chunks=20
        )
        
        # save filtered questions to file
        with open(FILTERED_QUESTIONS_JSON, 'w') as f:
            json.dump(filtered, f, indent=4)

        print(f"Filtered questions to {len(filtered)} items based on scenegraph and hypernyms.")
    else:
        # load filtered questions from file
        filtered = read_json(FILTERED_QUESTIONS_JSON)
        print(f"Loaded {len(filtered)} filtered questions from {FILTERED_QUESTIONS_JSON}.")
    
    # STEP 3: downsampling
    if not Path(DOWNSAMPLING_QUESTIONS_JSON).exists():
        downsampled = downsample_questions(
            questions=filtered,
            max_sample_size=40,
            random_seed=seed
        )
        print(f"Downsampled questions to {len(downsampled)} items.")

        # Save the downsampled questions
        with open(DOWNSAMPLING_QUESTIONS_JSON, 'w') as f:
            json.dump(downsampled, f, indent=4)

    else:
        # Load the downsampled questions from file
        print(f"Loading downsampled questions from {DOWNSAMPLING_QUESTIONS_JSON}.")
        downsampled = read_json(DOWNSAMPLING_QUESTIONS_JSON)

    # STEP 4: generate scenegraphs
    if not Path(SCENE_DESCRIPTIONS_JSON).exists():
        imgIds = sorted({v['imageId'] for v in downsampled.values()})
        scene_texts = generate_scene_description(
            scenegraph=scenegraph,
            image_ids=imgIds,
            mass_nouns=mass_nouns,
            gqa_lemmas=gqa_lemmas,
            random_seed=seed
        )
        # save scenegraphs
        with open(SCENE_DESCRIPTIONS_JSON, 'w') as f:
            json.dump(scene_texts, f, indent=4)
    else:
        # Load the scene descriptions from file
        print(f"Loading scene descriptions from {SCENE_DESCRIPTIONS_JSON}.")
        scene_texts = read_json(SCENE_DESCRIPTIONS_JSON)

    print(f"Generated scene descriptions for {len(scene_texts)} images.")

    # check if image_prompts_df and text_prompts_df already exist
    if not Path(IMAGE_PROMPTS_TSV).exists() and Path(TEXT_PROMPTS_TSV).exists():
       
        # STEP 5: build prompts
        scene_texts = format_scene_description(scene_texts)
        image_prompts_df = prompt_builder(
            downsampled,
            scene_texts,
            image_as_modality=True
        )
        text_prompts_df = prompt_builder(
            downsampled,
            scene_texts,
            image_as_modality=False
        )

        args = text_prompts_df['original_arg'].unique().tolist()
        print(len(args), "unique args in text prompts")
        # filter question types
        image_prompts_df = image_prompts_df[image_prompts_df['question_type'].isin(ALL_TYPES)]
        text_prompts_df = text_prompts_df[text_prompts_df['question_type'].isin(ALL_TYPES)]
        print(f"Built prompts for {len(text_prompts_df)} questions.")

        # save csv files to the output directory
        image_prompts_df.to_csv(IMAGE_PROMPTS_TSV, sep=',', index=False)
        text_prompts_df.to_csv(TEXT_PROMPTS_TSV, sep=',', index=False)
        end_time = time.time()
        print(f"Pipeline completed in {end_time - start_time:.2f} seconds.") # took 2.62 hrs for one run 
    else:
        # Load the text prompts from the TSV files
        print(f"Loading image prompts from {IMAGE_PROMPTS_TSV} and text prompts from {TEXT_PROMPTS_TSV}.")
        text_prompts_df = pd.read_csv(TEXT_PROMPTS_TSV, sep=',')
        # you could also load image_prompts_df here if needed

    # check if negative sampling files exist
    if not Path(NEG_LEAF_JSON).exists() or not Path(NEG_NON_LEAF_JSON).exists():
        print("Negative sampling files do not exist, generating them.")
        # STEP 6: negative sampling
        neg_leaf, neg_non_leaf = get_proper_args_for_negative_sampling(scenegraph, hyper_tree, text_prompts_df)
        # save neg_leaf, neg_non_leaf to data_path
    
        with open(NEG_LEAF_JSON, 'w') as f:
            json.dump(neg_leaf, f, indent=4)
        with open(NEG_NON_LEAF_JSON, 'w') as f:
            json.dump(neg_non_leaf, f, indent=4)
    else:
        # Load the negative sampling files
        print(f"Loading negative sampling files from {NEG_LEAF_JSON} and {NEG_NON_LEAF_JSON}.")
        neg_leaf = read_json(NEG_LEAF_JSON)
        neg_non_leaf = read_json(NEG_NON_LEAF_JSON)
  
    final_df_text = negative_sampling(mass_nouns, gqa_lemmas, NEGATIVE_SAMPLES_DIR, NEG_LEAF_JSON, NEG_NON_LEAF_JSON, ATTRIBUTE_OBJECT_MAP_PATH, RELATION_OBJECT_MAP_PATH, RELATION_SUBJECT_MAP_PATH, TEXT_PROMPTS_TSV, seed=seed)
    final_df_image = negative_sampling(mass_nouns, gqa_lemmas, NEGATIVE_SAMPLES_DIR, NEG_LEAF_JSON, NEG_NON_LEAF_JSON, ATTRIBUTE_OBJECT_MAP_PATH, RELATION_OBJECT_MAP_PATH, RELATION_SUBJECT_MAP_PATH, IMAGE_PROMPTS_TSV, seed=seed)
     # save final file to path
    final_df_image.to_csv(FINAL_DATA_FILE_PATH_IMAGE, sep=',', index=False)
    final_df_text.to_csv(FINAL_DATA_FILE_PATH_TEXT, sep=',', index=False)
    print(f"Final data saved to {FINAL_DATA_FILE_PATH_IMAGE} and {FINAL_DATA_FILE_PATH_TEXT}.")

if __name__ == "__main__":
    main()