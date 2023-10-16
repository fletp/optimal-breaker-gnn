"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""
import pandas as pd
import networkx as nx

def build_base_network(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Build the base network from the universal node and edge specifications."""
    G = nx.from_pandas_edgelist(
        edges,
        source="busbar_u",
        target="busbar_v",
        edge_attr=True
    )
    nodes = nodes.set_index("busbar_id")
    nx.set_node_attributes(
        G=G,
        values=nodes.to_dict(orient="index")
    )
    return G

def create_network_scenario(G: nx.Graph) -> nx.Graph:
    pass