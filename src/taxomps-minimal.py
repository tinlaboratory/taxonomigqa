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
    for i, batch in enumerate(tqdm(eval_set)):
        item = batch["item"]

        question = apply_template(batch["question"], lm, instruct, vision)

        if vlmscorer:
            dist = lm.next_word_distribution(question, image=None)
        else:
            dist = lm.next_word_distribution(question)

        probs, ranks = lm.query(dist, queries=[OPTIONS] * len(question))

        labels = [OPTIONS[i] for i in torch.tensor(probs).argmax(1).tolist()]

        # yes relative probs
        rel_probs = p_yes(probs)

        for j, l, p in zip(item, labels, rel_probs):
            results.append((j, l, p))


    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    utils.write_csv(
        results,
        path=f"{output_dir}/{model_name}.csv",
        header=["item", "label", "p_yes"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument(
        "--eval_path",
        type=str,
        default="data/gqa_entities/taxomps-hypernym.csv",
    )
    parser.add_argument("--output_dir", type=str, default="data/results/taxomps-ns-qa")
    parser.add_argument("--vlmscorer", "-v", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--instruct", action="store_true")
    args = parser.parse_args()

    main(args)