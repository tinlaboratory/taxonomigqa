# pipeline/config.py
from pathlib import Path

# 1) Compute project root by walking up from this file:
#    config.py lives at
#    multimodal-representations/src/preprocessing/my_data_prep/pipeline/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[4]
GQA_DATA_ROOT = Path(__file__).resolve().parents[5]


# 2) Now build all your paths relative to PROJECT_ROOT:
GQA_DATA        = GQA_DATA_ROOT / "data" / "gqa" / "data"          # .../data/gqa/data
ENTITY_TREE_DIR = PROJECT_ROOT / "data" / "gqa_entities"
FINAL_INFERENCE_DATA_DIR = PROJECT_ROOT / "data" / "behavioral-data" 
PREP_DATA       = PROJECT_ROOT / "src" / "preprocessing" / "my_data_prep" / "data"

# 3) Specific files
VAL_QUESTIONS          = GQA_DATA       / "val_all_questions.json"
VAL_SCENEGRAPHS        = GQA_DATA       / "val_sceneGraphs.json"
HYPERNYM_PATH          = ENTITY_TREE_DIR/ "noun-hypernyms.json"

MASS_NOUNS_JSON        = PREP_DATA      / "mass_nouns.json"
GQA_LEMMAS_CSV         = PREP_DATA      / "gqa-lemmas-preannotated.csv"
IMAGE_PROMPTS_TSV      = PREP_DATA      / "image_prompts.tsv"
TEXT_PROMPTS_TSV       = PREP_DATA      / "text_prompts.tsv"
NEG_LEAF_JSON          = PREP_DATA      / "neg_leaf.json"
NEG_NON_LEAF_JSON      = PREP_DATA      / "neg_non_leaf.json"
NEGATIVE_SAMPLES_DIR   = PREP_DATA      / "negative_sampling"

ATTRIBUTE_OBJECT_MAP_PATH   = PREP_DATA      / "merged_attribute_object_map.json"
RELATION_SUBJECT_MAP_PATH   = PREP_DATA      / "merged_relation_subject_map.json"
RELATION_OBJECT_MAP_PATH    = PREP_DATA      / "merged_relation_object_map.json"

# 4) Intermediate files
FILTERED_QUESTIONS_JSON = PREP_DATA     / "filtered_questions.json"
DOWNSAMPLING_QUESTIONS_JSON = PREP_DATA / "downsampling.json"
SCENE_DESCRIPTIONS_JSON = PREP_DATA     / "scenegraph_descriptions.json"

# 5) Final data files
FINAL_DATA_FILE_PATH_IMAGE = FINAL_INFERENCE_DATA_DIR / "model_inference_input_image.tsv"
FINAL_DATA_FILE_PATH_TEXT = FINAL_INFERENCE_DATA_DIR / "model_inference_input_text.tsv"

QUESTION_TYPE_GROUPS = {
    "EXIST":     ["exist"],
    "MATERIAL":  ["existMaterial", "existMaterialC", "existMaterialNot", "existMaterialNotC"],
    "ATTRIBUTE": [
        "existAttrNotC", "existAttr", "existAttrC", "existAttrNot",
        "existThat", "existThatC", "existThatNot", "existThatNotC"
    ],
    "RELATION": ['existRelS', 'existRelSC', 'existRelSRC']
}