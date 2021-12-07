import gc
import itertools
import random
import time
from string import Template
import gqlalchemy
import numpy as np
import dataset_parser
from sklearn.model_selection import train_test_split
from typing import Iterator, Dict, List, Tuple, Any

from gqlalchemy import Match



edge_remove_template = Template(
    'MATCH (a:$node_a_name{id: $node_a_id})-[edge]-(b:$node_b_name{id: $node_b_id}) DELETE edge;')

# NODE2VEC PARAMS
is_directed: bool = False
p = 1  # return parameter
q = 1 / 256  # in-out parameter
num_walks = 10
walk_length = 80
vector_size = 100
alpha = 0.02
window = 5
min_count = 1
seed = int(time.time())
workers = 4
min_alpha = 0.0001
sg = 1
hs = 0
negative = 5
epochs = 5

# Precision@k params
PRECISION_AT_K_CONST = 2 ** 8

memgraph = gqlalchemy.Memgraph("127.0.0.1", 7687)
def call_a_query_and_fetch(query: str) -> Iterator[Dict[str, Any]]:
    return memgraph.execute_and_fetch(query)


def call_a_query(query: str) -> None:
    memgraph.execute(query)


def set_node_embeddings() -> None:
    call_a_query("""CALL node2vec.set_embeddings({is_directed},{p}, {q}, {num_walks}, {walk_length}, {vector_size}, 
    {alpha}, {window}, {min_count}, {seed}, {workers}, {min_alpha}, {sg}, {hs}, {negative}) YIELD *""".format(
        is_directed=is_directed, p=p, q=q, num_walks=num_walks, walk_length=walk_length, vector_size=vector_size,
        alpha=alpha, window=window, min_count=min_count, seed=seed, workers=workers, min_alpha=min_alpha, sg=sg,
        hs=hs, negative=negative))


def get_all_edges() -> List[Tuple[gqlalchemy.Node, gqlalchemy.Node]]:
    results = Match() \
        .node(dataset_parser.NODE_NAME, variable="node_a") \
        .to(dataset_parser.EDGE_NAME, variable="edge") \
        .node(dataset_parser.NODE_NAME, variable="node_b") \
        .execute()

    return [(result["node_a"], result["node_b"]) for result in results]


def remove_edges(edges: List[Tuple[gqlalchemy.Node, gqlalchemy.Node]]) -> None:
    queries = [edge_remove_template.substitute(node_a_name=dataset_parser.NODE_NAME,
                                               node_a_id=edge[0].properties["id"],
                                               node_b_name=dataset_parser.NODE_NAME,
                                               node_b_id=edge[1].properties["id"]) for edge in edges]
    for query in queries:
        call_a_query(query)


def split_edges_train_test(edges: List[Tuple[gqlalchemy.Node, gqlalchemy.Node]], test_size: float = 0.2) -> (
        List[Tuple[gqlalchemy.Node, gqlalchemy.Node]], List[Tuple[gqlalchemy.Node, gqlalchemy.Node]]):
    edges_train, edges_test = train_test_split(edges, test_size=test_size, random_state=int(time.time()))

    return edges_train, edges_test


def get_embeddings_as_properties():
    embeddings: Dict[int, List[float]] = {}
    results = Match() \
        .node(dataset_parser.NODE_NAME, variable="node") \
        .execute()

    for result in results:
        node: gqlalchemy.Node = result["node"]
        if not "embedding" in node.properties:
            continue
        embeddings[node.properties["id"]] = node.properties["embedding"]

    return embeddings


def calculate_adjacency_matrix(embeddings: Dict[int, List[float]], threshold=0.0) -> Dict[Tuple[int, int], float]:
    def get_edge_weight(i, j) -> float:
        return np.dot(embeddings[i], embeddings[j])

    nodes = list(embeddings.keys())
    nodes = sorted(nodes)
    adj_mtx_r = {}
    cnt = 0
    for pair in itertools.combinations(nodes, 2):

        if cnt % 1000000 == 0:
            adj_mtx_r = {k: v for k, v in sorted(adj_mtx_r.items(), key=lambda item: -1 * item[1])}
            adj_mtx_r = {k: adj_mtx_r[k] for k in list(adj_mtx_r)[:3*PRECISION_AT_K_CONST]}
            gc.collect()

        if cnt % 10000 == 0:
            print(cnt)

        weight = get_edge_weight(pair[0], pair[1])
        if weight <= threshold:
            continue
        cnt += 1
        adj_mtx_r[(pair[0], pair[1])] = get_edge_weight(pair[0], pair[1])

    return adj_mtx_r


def compute_precision_at_k(predicted_edges: Dict[Tuple[int, int], float],
                           test_edges: Dict[Tuple[int, int], int], max_k):
    precision_scores = []  # precision at k
    delta_factors = []
    correct_edge = 0
    count = 0
    for edge in predicted_edges:
        if count > max_k:
            break

        # if our guessed edge is really in graph
        # this is due representation problem: (2,1) edge in undirected graph is saved in memory as (2,1)
        # but in adj matrix it is calculated as (1,2)
        if edge in test_edges or (edge[1], edge[0]) in test_edges:
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (count + 1))  # (number of correct guesses) / (number of attempts)
        count += 1

    return precision_scores, delta_factors


def compute_MAP(predicted_edges: Dict[Tuple[int, int], float],
                test_edges: Dict[Tuple[int, int], int], max_k) -> float:
    node_edges: Dict[int, Dict[Tuple[int, int], float]] = {}

    for source, target in predicted_edges:
        if source not in node_edges:
            node_edges[source] = {}
        if target not in node_edges:
            node_edges[target] = {}

        node_edges[source][(source, target)]=predicted_edges[(source,target)]
        node_edges[target][(target, source)]=predicted_edges[(source,target)]

    node_num = len(node_edges)
    node_AP = [0.0] * len(node_edges)
    count = 0
    for node in node_edges:

        precision_scores, delta_factors = compute_precision_at_k(node_edges[node],
                                                                 test_edges, max_k)
        precision_rectified = [p * d for p, d in zip(precision_scores, delta_factors)]
        if (sum(delta_factors) == 0):
            node_AP[count] = 0
        else:
            node_AP[count] = float(sum(precision_rectified) / sum(delta_factors))
        count += 1
    return sum(node_AP) / count


def import_dataset(filename: str) -> None:
    with open(filename, "r") as filestream:
        queries = filestream.readlines()
    for query in queries:
        call_a_query(query)


def main():
    print("Cleaning database...")
    call_a_query("MATCH (n) DETACH DELETE (n);")

    print("Importing dataset...")
    import_dataset("query.cypherl")

    print("Getting all edges...")
    edges = get_all_edges()
    print("Current number of edges is {}".format(len(edges)))

    print("Splitting edges in train, test group...")
    edges_train, edges_test = split_edges_train_test(edges=edges, test_size=0.2)
    print("Splitting edges done.")

    print("Removing edges from graph.")
    remove_edges(edges_test)
    print("Edges removed.")
    train_edges_dict = {(node_from.properties["id"], node_to.properties["id"]): 1 for node_from, node_to in edges_train}
    test_edges_dict = {(node_from.properties["id"], node_to.properties["id"]): 1 for node_from, node_to in edges_test}

    # Calculate and get node embeddings
    print("Setting node embeddings as graph property...")
    set_node_embeddings()
    print("Embedding for every node set.")

    node_emeddings = get_embeddings_as_properties()

    # Calculate adjacency matrix
    print("Calculating adjacency matrix from embeddings.")
    adj_matrix = calculate_adjacency_matrix(embeddings=node_emeddings, threshold=0.0)
    print("Adjacency matrix calculated")
    # print(adj_matrix)

    print("Getting predicted edges...")
    predicted_edge_list = adj_matrix
    print("Predicted edge list is of length:", len(predicted_edge_list), "\n")

    print("Sorting predicted edge list")

    # We need to sort predicted edges so that ones that are most likely to appear are first in list
    sorted_predicted_edges = {k: v for k, v in sorted(predicted_edge_list.items(), key=lambda item: -1 * item[1])}
    print("Predicted edges sorted...")

    print("Filtering predicted edges that are not in train list...")
    # taking only edges that we are predicting to appear, not ones that are already in graph
    sorted_predicted_edges = {k: v for k, v in sorted_predicted_edges.items() if k not in train_edges_dict}

    print("Calculating precision@k...")
    precision_scores, delta_factors = compute_precision_at_k(predicted_edges=sorted_predicted_edges,
                                                             test_edges=test_edges_dict,
                                                             max_k=PRECISION_AT_K_CONST)
    print("precision score", precision_scores)

    with open("../results.txt", 'a+') as fh:
        fh.write(" ".join(str(precision) for precision in precision_scores))
        fh.write("\n")


if __name__ == '__main__':
    main()
