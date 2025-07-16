import json
from semantic_memory import taxonomy
from collections import defaultdict

def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_tree(path):
    noun_hypernyms = read_json(path)

    # get all hypernym paths of a noun.
    # the hypernyms list is structured so that succeeding elements are more general than the previous ones
    # example if the entry is "jacket": ["coat", "garment", "clothing"], then "jacket" is a type of "coat", "coat" is a type of "garment", and "garment" is a type of "clothing"

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
    return Tree

