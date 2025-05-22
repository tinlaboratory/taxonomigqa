import json
import networkx as nx

import torch
from sklearn.covariance import ledoit_wolf

import gc
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
from tqdm import tqdm


def get_categories(noun_or_verb="noun", model_name="gemma"):

    cats = {}
    if noun_or_verb == "noun":
        with open(f"data/rsa-graphs/noun_synsets_wordnet_{model_name}.json", "r") as f:
            for line in f:
                cats.update(json.loads(line))
        G = nx.read_adjlist(
            f"data/rsa-graphs/noun_synsets_wordnet_hypernym_graph_{model_name}.adjlist",
            create_using=nx.DiGraph(),
        )
    elif noun_or_verb == "verb":
        with open(f"data/rsa-graphs/verb_synsets_wordnet_{model_name}.json", "r") as f:
            for line in f:
                cats.update(json.loads(line))
        G = nx.read_adjlist(
            f"data/rsa-graphs/verb_synsets_wordnet_hypernym_graph_{model_name}.adjlist",
            create_using=nx.DiGraph(),
        )

    cats = {k: list(set(v)) for k, v in cats.items() if len(set(v)) > 50}
    G = nx.DiGraph(G.subgraph(cats.keys()))

    reversed_nodes = list(reversed(list(nx.topological_sort(G))))
    for node in reversed_nodes:
        children = list(G.successors(node))
        if len(children) == 1:
            child = children[0]
            parent_lemmas_not_in_child = set(cats[node]) - set(cats[child])
            if (
                len(list(G.predecessors(child))) == 1
                or len(parent_lemmas_not_in_child) < 5
            ):
                grandchildren = list(G.successors(child))
                for grandchild in grandchildren:
                    G.add_edge(node, grandchild)
                G.remove_node(child)

    G = nx.DiGraph(G.subgraph(cats.keys()))
    sorted_keys = list(nx.topological_sort(G))
    cats = {k: cats[k] for k in sorted_keys}

    return cats, G, sorted_keys


def category_to_indices(category, vocab_dict):
    return [vocab_dict[w] for w in category]


def get_words_sim_to_vec(query: torch.tensor, unembed, vocab_list, k=300):
    similar_indices = torch.topk(unembed @ query, k, largest=True).indices.cpu().numpy()
    return [vocab_list[idx] for idx in similar_indices]


def estimate_single_dir_from_embeddings(category_embeddings):
    category_mean = category_embeddings.mean(dim=0)

    cov = ledoit_wolf(category_embeddings.cpu().numpy())
    cov = torch.tensor(cov[0], device=category_embeddings.device)
    pseudo_inv = torch.linalg.pinv(cov)
    lda_dir = pseudo_inv @ category_mean
    lda_dir = lda_dir / torch.norm(lda_dir)
    lda_dir = (category_mean @ lda_dir) * lda_dir

    return lda_dir, category_mean


def estimate_cat_dir(category_lemmas, unembed, vocab_dict):
    category_embeddings = unembed[category_to_indices(category_lemmas, vocab_dict)]
    # print("The stuff below this apparently takes a lot of time:")
    lda_dir, category_mean = estimate_single_dir_from_embeddings(category_embeddings)

    return {"lda": lda_dir, "mean": category_mean}


import inflect

p = inflect.engine()


def noun_to_gemma_vocab_elements(word, vocab_set):
    word = word.lower()
    plural = p.plural(word)
    add_cap_and_plural = [word, word.capitalize(), plural, plural.capitalize()]
    add_space = ["▁" + w for w in add_cap_and_plural]
    return vocab_set.intersection(add_space)


def get_animal_category(data, categories, vocab_dict, g):
    vocab_set = set(vocab_dict.keys())

    animals = {}
    animals_ind = {}
    animals_g = {}
    animals_token = {}

    for category in categories:
        animals[category] = []
        animals_ind[category] = []
        animals_g[category] = []
        animals_token[category] = []

    for category in categories:
        lemmas = data[category]
        for w in lemmas:
            animals[category].extend(noun_to_gemma_vocab_elements(w, vocab_set))

        for word in animals[category]:
            animals_ind[category].append(vocab_dict[word])
            animals_token[category].append(word)
            animals_g[category] = g[animals_ind[category]]
    return animals_token, animals_ind, animals_g


def get_gamma(MODEL_NAME, device, v2s=False):
    if v2s:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, device_map=device
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, device_map=device
        )

    gamma = model.get_output_embeddings().weight.detach()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return gamma


# Load the g tensor
def load_g_tensor(filename, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tensor
    g = torch.load(filename, map_location=device)

    print(f"Tensor loaded from {filename}")
    return g


def get_g(MODEL_NAME, device, v2s=False):
    # gamma = get_gamma(MODEL_NAME, device)
    # model_short_name = MODEL_NAME.split('/')[-1]
    # gamma = load_g_tensor("./saved_tensors/"+ model_short_name + "_g_tensor.pt", device)
    gamma = get_gamma(MODEL_NAME, device, v2s)

    W, d = gamma.shape
    gamma_bar = torch.mean(gamma, dim=0)
    centered_gamma = gamma - gamma_bar

    Cov_gamma = centered_gamma.T @ centered_gamma / W
    eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
    inv_sqrt_Cov_gamma = (
        eigenvectors @ torch.diag(1 / torch.sqrt(eigenvalues)) @ eigenvectors.T
    )
    sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    g = centered_gamma @ inv_sqrt_Cov_gamma

    return g, inv_sqrt_Cov_gamma, sqrt_Cov_gamma


def get_vocab(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_dict = tokenizer.get_vocab()
    vocab_list = [None] * (max(vocab_dict.values()) + 1)
    for word, index in vocab_dict.items():
        vocab_list[index] = word

    return vocab_dict, vocab_list


def compute_lambdas(texts, MODEL_NAME, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="auto"
    )

    assert (
        tokenizer.padding_side == "left"
    ), "The tokenizer padding side must be 'left'."

    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        lambdas = outputs.hidden_states[-1][:, -1, :]

    return lambdas
