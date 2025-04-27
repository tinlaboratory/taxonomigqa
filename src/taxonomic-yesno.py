import argparse
import os
import pathlib
import torch
import utils

from minicons import scorer
from string import Template
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

OPTIONS = ["Yes", "No", "yes", "no"]


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


def apply_template(questions, lm, instruct, vision):
    stimuli = [
        chat_template(
            q,
            lm.tokenizer,
            "Answer the question either with Yes or No.",
            instruct,
            vision,
        )
        for q in questions
    ]
    return stimuli


def p_yes(probs):
    alls = torch.tensor(probs).sum(1)
    yeses = [[p[0], p[2]] for p in probs]
    return (torch.tensor(yeses).sum(1) / alls).tolist()


def main(args):
    model = args.model
    vlmscorer = args.vlmscorer
    model_name = model.replace("/", "_")

    eval_path = args.eval_path
    output_dir = args.output_dir

    if vlmscorer:
        vision = True
        lm = scorer.VLMScorer(model, device=args.device)
    else:
        vision = False
        lm = scorer.IncrementalLMScorer(
            model, device=args.device, trust_remote_code=True
        )

    instruct = args.instruct

    if model == "meta-llama/Llama-3.2-11B-Vision":
        instruct = Template(
            f"{lm.tokenizer.tokenizer.bos_token}$system $sentence Answer:"
        )

    eval_data = utils.read_csv_dict(eval_path)
    eval_set = DataLoader(eval_data, batch_size=args.batch_size)

    results = []
    for batch in tqdm(eval_set):
        idx = batch["idx"]
        item = batch["item"]
        hypernym_question = apply_template(
            batch["hypernym_question"], lm, instruct, vision
        )
        negative_question = apply_template(
            batch["negative_question"], lm, instruct, vision
        )
        swapped_question = apply_template(
            batch["swapped_question"], lm, instruct, vision
        )

        if vlmscorer:
            hypernym_dist = lm.next_word_distribution(hypernym_question, image=None)
            negative_dist = lm.next_word_distribution(negative_question, image=None)
            swapped_dist = lm.next_word_distribution(swapped_question, image=None)
        else:
            hypernym_dist = lm.next_word_distribution(hypernym_question)
            negative_dist = lm.next_word_distribution(negative_question)
            swapped_dist = lm.next_word_distribution(swapped_question)

        hypernym_probs, hypernym_ranks = lm.query(
            hypernym_dist, queries=[OPTIONS] * len(hypernym_question)
        )
        negative_probs, negative_ranks = lm.query(
            negative_dist, queries=[OPTIONS] * len(negative_question)
        )
        swapped_probs, swapped_ranks = lm.query(
            swapped_dist, queries=[OPTIONS] * len(swapped_question)
        )

        hypernym_labels = [
            OPTIONS[i] for i in torch.tensor(hypernym_probs).argmax(1).tolist()
        ]
        negative_labels = [
            OPTIONS[i] for i in torch.tensor(negative_probs).argmax(1).tolist()
        ]
        swapped_labels = [
            OPTIONS[i] for i in torch.tensor(swapped_probs).argmax(1).tolist()
        ]

        # hypernum scores
        hypernym_p_yes = p_yes(hypernym_probs)
        negative_p_yes = p_yes(negative_probs)
        swapped_p_yes = p_yes(swapped_probs)

        # hypernym_scores = lm.sequence_score(hypernym_sentences)
        # negative_scores = lm.sequence_score(negative_sentences)

        for j, i, h, n, s, hy, ny, sy in zip(
            item,
            idx,
            hypernym_labels,
            negative_labels,
            swapped_labels,
            hypernym_p_yes,
            negative_p_yes,
            swapped_p_yes,
        ):
            results.append((j, i, h, n, s, hy, ny, sy))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    utils.write_csv(
        results,
        path=f"{output_dir}/{model_name}.csv",
        header=[
            "item",
            "idx",
            "hypernym_pred",
            "negative_pred",
            "swapped_pred",
            "hypernym_yes",
            "negative_yes",
            "swapped_yes",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--eval_path",
        type=str,
        default="data/things-taxonomic-sensitivity/things-hypernym-minimal-pairs-qa.csv",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/results/hypernym-minimal-pairs-qa"
    )
    parser.add_argument("--vlmscorer", "-v", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--instruct", action="store_true")
    args = parser.parse_args()

    main(args)
