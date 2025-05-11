import argparse
import csv
import pathlib
import os
import torch
import utils

from collections import defaultdict
from copy import deepcopy
from minicons import cwe
from ordered_set import OrderedSet
from string import Template
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def chat_template(sentence, tok, system=None, instruct=True, vision=False):
    """
    A function that applies the model's chat template to simulate
    an interaction environment.
    """
    if instruct == True:
        if system is None:
            if not vision:
                return tok.apply_chat_template(
                    [{"role": "user", "content": sentence}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                return tok.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": sentence}]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            try:
                if not vision:
                    return tok.apply_chat_template(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": sentence},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    try:
                        return tok.apply_chat_template(
                            [
                                {
                                    "role": "system",
                                    "content": [{"type": "text", "text": system}],
                                },
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": sentence}],
                                },
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    except:
                        try:
                            # print("second error")
                            return tok.apply_chat_template(
                                [
                                    {
                                        "role": "system",
                                        "content": system,
                                    },
                                    {
                                        "role": "user",
                                        "content": [{"type": "text", "text": sentence}],
                                    },
                                ],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        except:
                            # print("third error")
                            return tok.apply_chat_template(
                                [
                                    {
                                        "role": "user",
                                        "content": f"{system} {sentence}",
                                    },
                                ],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
            except:
                return tok.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"{system} {sentence}"}
                            ],
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
    elif isinstance(instruct, Template):
        return instruct.substitute(system=system, sentence=sentence)
    else:
        return f"{system} {sentence} Answer:"


def main(args):

    model = args.model
    model_name = model.replace("/", "_")

    vlm = args.vlm

    # if vlm:
    #     lm = cwe.VisualCWE(model, device=args.device)
    # else:
    lm = cwe.CWE(model, device=args.device)

    # questions = utils.read_csv_dict(
    #     "data/gqa_dataset/contextualized_text_analysis_data.tsv", True
    # )

    # questions_filtered = [q for q in questions if q["substitution_hop"] != "0"]

    data_path = "data/gqa_entities"

    category_pairs_path = f"{data_path}/category-membership.csv"
    lemma_path = f"{data_path}/gqa-lemmas-annotated.csv"

    lexicon = defaultdict(dict)
    lemmas = utils.read_csv_dict(lemma_path)
    lemmas = [l for l in lemmas if l["remove"] not in ["?", "1"]]
    for entry in lemmas:
        lexicon[entry["lemma"]] = entry

    lexicon["sports equiment"] = {
        "lemma": "sports equiment",
        "singular": "sports equiment",
        "plural": "sports equiment",
        "article": "sports equiment",
        "taxonomic_phrase": "are a type of",
        "novel_property_is": "",
        "novel_property_have": "",
        "generic": "p",
        "remove": "",
    }

    lexicon = dict(lexicon)

    data = utils.read_csv_dict("data/gqa_dataset/merged_model_results.csv", tsv=True)
    model_ids = defaultdict(list)

    ids = utils.read_csv_dict("data/gqa_dataset/qwen-base-correct.csv")
    for entry in ids:
        model_ids[entry["model"]].append(entry["question_id"])

    question_types = [
        "existAttrC",
        "existAttrNotC",
        "existMaterialC",
        "existMaterialNotC",
    ]

    if vlm:
        target = "vlm"
        vision = True
        model_type = "vision_text"
        model_id = "vlm_text_qwen2.5VL"
    else:
        target = "lm"
        vision = False
        model_type = "text_only"
        model_id = "vlm_text_qwen2.5VL"

    exists_data = [
        d
        for d in data
        if d["question_type"] in question_types
        and d["substitution_hop"] not in ["0", "-100"]
        and d["question_id"] in model_ids[target]
    ]

    concept_log = defaultdict(str)

    target_concepts_equal = [
        "couch",
        "pan",
        "wine",
        "donut",
        "woman",
        "football",
        "pouch",
        "surfer",
    ]
    for entry in target_concepts_equal:
        concept_log[entry] = "equal"

    # vlm > lm
    target_concepts_vlm_better = [
        "sofa",
        "mud",
        "cup",
        "oven",
        "baseball",
        "skateboard",
        "bear",
        "refrigerator",
    ]
    for entry in target_concepts_vlm_better:
        concept_log[entry] = "vlm-better"

    concept_log = dict(concept_log)

    target_concepts = target_concepts_equal + target_concepts_vlm_better

    questions_filtered = [
        e for e in exists_data if e["original_arg"] in target_concepts
    ]

    nots = []
    no_q = []
    no_q_hypo = []
    questions_filtered_clean = []
    nok = []

    no_q_sg = []
    no_q_pl = []

    for i, entry in enumerate(questions_filtered):
        question = entry["input"].split("Question:")[-1]
        new_entry = deepcopy(entry)
        hypernym = entry["hyper_form"]
        hyponym = entry["hypo_form"]

        hyper_arg = entry["argument"]

        singular_hyper = lexicon[hyper_arg]["singular"]
        plural_hyper = lexicon[hyper_arg]["plural"]

        if hypernym not in question:
            # no_q.append(entry)
            if hypernym == singular_hyper:
                if plural_hyper not in question:
                    if hyper_arg in question:
                        new_entry["hyper_form"] = hyper_arg
                    else:
                        no_q_sg.append(entry)
                else:
                    new_entry["hyper_form"] = plural_hyper
            elif hypernym == plural_hyper:
                if singular_hyper not in question:
                    no_q_pl.append(entry)
                else:
                    new_entry["hyper_form"] = singular_hyper
            elif hyper_arg in question:
                new_entry["hyper_form"] = hyper_arg
            else:
                no_q.append(entry)

        if hyponym not in entry["input"].split("Question:")[0]:
            no_q_hypo.append(entry)
            # unpossess

        if entry["argument"] not in lexicon.keys():
            nok.append(entry)

        new_entry["input"] = chat_template(entry["input"], lm.tokenizer, vision=vision)
        new_entry["idx"] = i
        new_entry["subset"] = concept_log[entry["original_arg"]]

        questions_filtered_clean.append(new_entry)

    utils.write_csv_dict(
        f"data/token-analysis/token-analysis_data-{model_name}.csv",
        questions_filtered_clean,
    )

    # check

    noq = []
    for entry in questions_filtered_clean:
        question = entry["input"].split("Question:")[-1]
        # new_entry = deepcopy(entry)
        hypernym = entry["hyper_form"]
        if not (hypernym in question):
            noq.append(entry)

    assert len(noq) == 0

    # batches = DataLoader(questions_filtered_clean, batch_size=args.batch_size)

    # # layerwise_reps = defaultdict(list)
    # layerwise = defaultdict(list)
    # for batch in tqdm(batches):
    #     queries = list(zip(batch["input"], batch["hypo_form"], batch["hyper_form"]))
    #     reps = lm.extract_paired_representations(
    #         queries, layer="all", multi_strategy="all"
    #     )
    #     # layerwise_reps[layer]
    #     for layer in range(lm.layers + 1):
    #         # layerwise_reps[layer].append((reps[0][layer], reps[1][layer]))
    #         cosines = [
    #             torch.cosine_similarity(x1, x2[-1].unsqueeze(0)).max().item()
    #             for x1, x2 in zip(reps[0][layer], reps[1][layer])
    #         ]
    #         layerwise[layer].extend(cosines)

    # results = []

    # for l, sims in layerwise.items():
    #     for i, sim in enumerate(sims):
    #         results.append((i, l, sim))

    # pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    # utils.write_csv(results, f"{args.output_dir}/{model_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_dir", type=str, default="data/results/gqa-cwe-sims")
    parser.add_argument("--vlm", "-v", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)
