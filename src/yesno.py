import argparse
import pathlib

from minicons import scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils


def chat_template(sequence, tokenizer=None):
    formatted = [{"role": "user", "content": sequence.strip()}]
    return tokenizer.apply_chat_template(
        formatted, tokenize=False, add_generation_prompt=True
    )


def main(args):
    model = args.model
    model_name = model.replace("/", "_")
    stimuli_path = args.stimuli_path

    lm = scorer.IncrementalLMScorer(model, device=args.device, torch_dtype="bfloat16")

    stimuli = utils.read_csv_dict(f"{stimuli_path}/stimuli.csv")

    dl = DataLoader(stimuli, batch_size=args.batch_size, shuffle=False)

    results = []
    for batch in tqdm(dl, desc="Batches"):
        if args.chat:
            question = [chat_template(q, lm.tokenizer) for q in batch["question"]]
            sep = ""
        else:
            question = batch["question"]
            sep = " "

        yes_scores = lm.conditional_score(
            question, ["Yes"] * len(question), separator=sep
        )
        no_scores = lm.conditional_score(
            question, ["No"] * len(question), separator=sep
        )

        for i, (y, n) in enumerate(zip(yes_scores, no_scores)):
            results.append(
                {
                    "concept1": batch["concept1"][i],
                    "concept2": batch["concept2"][i],
                    "yes": y,
                    "no": n,
                }
            )

    pathlib.Path(f"{stimuli_path}/results").mkdir(parents=True, exist_ok=True)
    utils.write_csv_dict(f"{stimuli_path}/results/{model_name}.csv", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stimuli_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()

    main(args)
