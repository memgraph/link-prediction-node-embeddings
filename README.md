# Physics paper recommender

This project is part of tutorial on `Link prediction with node2vec`. 

For this to work, you will need:
1. The **[MAGE graph library](https://memgraph.com/docs/mage/installation)**
2. **[Memgraph Lab](https://memgraph.com/product/lab)** - an application for querying Memgraph and visualizing graphs
3. **[gqlalchemy](https://github.com/memgraph/gqlalchemy)** - a Python driver and object graph mapper (OGM)



## Dataset parser
In order to parse Collaboration dataset, use  `public/dataset_parser.py`. It assumes existance of file `CA-HepPh.txt` in `root`.

In order to run it, use following command:
```bash
python3 public/dataset_parser.py
```

This will prepare cypher queries which will be used in `public/main.py`

## Link prediction script

Script in  `public/main.py` will do the following:
* Drop database
* Import dataset from file `query.cypherl` prepared with `public/dataset_parser.py`.
* Split edges from Memgraph into test and train set
* Remove test set edges from Memgraph
* Run node2vec to get node embeddings
* Make link predictions
* Append fresh precision@k in `results.txt`

```bash
python3 public/main.py
```

## Plotting results
In order to plot results use `public/main.py`

```bash
python3 public/plot.py
```
