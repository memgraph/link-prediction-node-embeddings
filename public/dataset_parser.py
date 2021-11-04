from string import Template
from typing import List, Tuple, Dict

FILENAME = "CA-HepPh.txt"
#FILENAME = "karate.txt"

NODE_NAME = "Collaborator"
EDGE_NAME = "COLLABORATED_WITH"

edge_template = Template(
    'MERGE (a:$node_name_a {id: $id_a}) MERGE (b:$node_name_b {id: $id_b}) CREATE (a)-[:$edge_name]->(b);')


def parse_edges_dataset(filename=FILENAME) -> List[Tuple[int, int]]:
    with open(filename) as file:
        lines = file.readlines()
    edges: Dict[Tuple[int, int]] = {}
    for line in lines:
        if line.startswith("#"):
            continue
        line = line.strip()
        line_parts = line.split("\t") ## change to tab
        edge = (int(line_parts[0]), int(line_parts[1]))
        if (edge[1], edge[0]) in edges:
            continue
        edges[edge] = 1

    return list(edges.keys())


def create_queries(edges: List[Tuple[int, int]]):
    queries: List[str] = ["CREATE INDEX ON :{node_name}(id);".format(node_name=NODE_NAME)]
    for source, target in edges:
        queries.append(edge_template.substitute(id_a=source,
                                                id_b=target,
                                                node_name_a=NODE_NAME,
                                                node_name_b = NODE_NAME,
                                                edge_name= EDGE_NAME))
    return queries


def main():
    edges = parse_edges_dataset()
    queries = create_queries(edges)
    print("\n".join(queries))

    file = open('query.cypherl', 'w')
    file.write("\n".join(queries))
    file.close()


if __name__ == '__main__':
    main()
