import argparse
import nltk
import pathlib

# nltk.download('wordnet')

import json
import networkx as nx
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer

import inflect


def main(args):
    model = args.model
    model_name = model.replace("/", "_")

    tokenizer = AutoTokenizer.from_pretrained(model)

    # model = "llava"
    # model = "molmo"
    # model = "vicuna"
    # model = "qwen2"
    # model = "llama-3.2-11B-vision"
    # model = "llama-3.1-8B"
    # model = "llama-3.1-8B-instruct"
    # model = "llava-onevision-qwen"
    # model = "llama-3.2-11B-vision-instruct"
    # model = "llava-1.6-mistral-7b"

    vocab = tokenizer.get_vocab()
    vocab_set = set(vocab.keys())

    p = inflect.engine()

    if "llava" in model_name or "vicuna" in model_name or "mistral" in model_name:
        tokenizer_specific_char = "▁"
    else:
        tokenizer_specific_char = "Ġ"

    # for Molmo, Qwen
    # tokenizer_specific_char = "▁" # for llava 1.5, vicuna-7b-v1.5
    # For gemma-2b: ▁

    def get_all_hyponym_lemmas(synset):
        hyponyms = synset.hyponyms()
        lemmas = set()
        for hyponym in hyponyms:
            lemmas.update(lemma.name() for lemma in hyponym.lemmas())
            lemmas.update(
                get_all_hyponym_lemmas(hyponym)
            )  # Recursively get lemmas from hyponyms,

        return lemmas

    all_noun_synsets = list(wn.all_synsets(pos=wn.NOUN))
    noun_lemmas = {}
    for s in all_noun_synsets:
        lemmas = get_all_hyponym_lemmas(s)
        # add and remove space bc of how gemma vocab works
        lemmas = vocab_set.intersection({tokenizer_specific_char + l for l in lemmas})
        noun_lemmas[s.name()] = {l[1:] for l in lemmas}

    large_nouns = {k: v for k, v in noun_lemmas.items() if len(v) > 5}

    # Construct the hypernym inclusion graph among large categories
    G_noun = nx.DiGraph()

    nodes = list(large_nouns.keys())
    for key in nodes:
        for path in wn.synset(key).hypernym_paths():
            # ancestors included in the cleaned set
            ancestors = [s.name() for s in path if s.name() in nodes]
            if len(ancestors) > 1:
                G_noun.add_edge(ancestors[-2], key)  # first entry is itself
            else:
                print(f"no ancestors for {key}")

    G_noun = nx.DiGraph(G_noun.subgraph(nodes))

    # if a node has only one child, and that child has only one parent, merge the two nodes
    def merge_nodes(G, lemma_dict):
        topological_sorted_nodes = list(reversed(list(nx.topological_sort(G))))
        for node in topological_sorted_nodes:
            children = list(G.successors(node))
            if len(children) == 1:
                child = children[0]
                parent_lemmas_not_in_child = lemma_dict[node] - lemma_dict[child]
                if (
                    len(list(G.predecessors(child))) == 1
                    or len(parent_lemmas_not_in_child) < 6
                ):
                    grandchildren = list(G.successors(child))

                    if len(parent_lemmas_not_in_child) > 1:
                        if len(grandchildren) > 0:
                            lemma_dict[node + ".other"] = parent_lemmas_not_in_child
                            G.add_edge(node, node + ".other")

                    # del synset_lemmas[child]
                    for grandchild in grandchildren:
                        G.add_edge(node, grandchild)
                    G.remove_node(child)
                    # print(f"merged {node} and {child}")

    merge_nodes(G_noun, large_nouns)
    large_nouns = {k: v for k, v in large_nouns.items() if k in G_noun.nodes()}

    # make a gemma specific version
    def _noun_to_gemma_vocab_elements(word):
        word = word.lower()
        plural = p.plural(word)
        add_cap_and_plural = [word, word.capitalize(), plural, plural.capitalize()]
        add_space = [tokenizer_specific_char + w for w in add_cap_and_plural]
        return vocab_set.intersection(add_space)
    
    pathlib.Path("data/rsa-graphs").mkdir(exist_ok=True, parents=True)

    with open(f"data/rsa-graphs/noun_synsets_wordnet_{model_name}.json", "w") as f:
        for synset, lemmas in large_nouns.items():
            gemma_words = []
            for w in lemmas:
                gemma_words.extend(_noun_to_gemma_vocab_elements(w))

            f.write(json.dumps({synset: gemma_words}) + "\n")

    nx.write_adjlist(
        G_noun, f"data/rsa-graphs/noun_synsets_wordnet_hypernym_graph_{model_name}.adjlist"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    main(args)
