"""Graph dataset adapter for SVAMP — identical structure to Gsm8kGraphDataset."""
import os
import glob
import pickle
import random

import networkx as nx
import torch
from sentence_transformers import SentenceTransformer

from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION

os.environ["TOKENIZERS_PARALLELISM"] = "false"


_SENTENCE_MODEL = None


def get_sentence_embedding(sentence):
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SENTENCE_MODEL.encode(sentence)


def precompute_role_embeddings(save_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    role_embeddings = {}
    for role, description in ROLE_DESCRIPTION.items():
        role_with_desc = f"{role}: {description.strip()}"
        role_embeddings[role] = torch.tensor(model.encode(role_with_desc))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(role_embeddings, f)
    print(f"Precomputed {len(role_embeddings)} SVAMP role embeddings → {save_path}")
    return role_embeddings


class SvampGraphDataset:
    def __init__(self, data_dir, sample_size=0, pretrain=True, data_dir_ef=None):
        self.data_dir = data_dir
        self.data_dir_ef = data_dir_ef
        self.pretrain = pretrain

        cache_path = os.path.join(self.data_dir, "precomputed_role_embeddings.pkl")
        if not os.path.exists(cache_path):
            self.precomputed_embeddings = precompute_role_embeddings(cache_path)
        else:
            with open(cache_path, "rb") as f:
                self.precomputed_embeddings = pickle.load(f)
            print(f"Loaded {len(self.precomputed_embeddings)} precomputed SVAMP role embeddings")

        self.graph_list = self._load_and_convert_graphs(sample_size)
        self.node_label_list = [0]
        self.edge_label_list = [0]

    def _load_and_convert_graphs(self, sample_size):
        if self.pretrain:
            graph_files = glob.glob(os.path.join(self.data_dir, "*.pt"))
        else:
            if not self.data_dir_ef:
                print("Warning: data_dir_ef not provided")
                return []
            graph_files = glob.glob(os.path.join(self.data_dir_ef, "*.pt"))

        print(f"Found {len(graph_files)} graph files")

        if self.pretrain:
            graph_files = [f for f in graph_files
                           if "True" in os.path.basename(f) or "solved" in os.path.basename(f).lower()]
            print(f"Filtered {len(graph_files)} successful graphs for pretraining")

        if not graph_files:
            print("Warning: No valid graph files found")
            return []

        if sample_size and sample_size > 0 and len(graph_files) > sample_size:
            random.seed(42)
            graph_files = random.sample(graph_files, sample_size)

        sorted_roles = sorted(ROLE_DESCRIPTION.keys())
        role_to_id = {role: i for i, role in enumerate(sorted_roles)}
        id_to_role = {i: role for i, role in enumerate(sorted_roles)}
        self.role_to_id = role_to_id
        self.id_to_role = id_to_role

        nx_graphs = []
        for file in graph_files:
            try:
                try:
                    pyg_graph = torch.load(file, weights_only=False)
                except TypeError:
                    pyg_graph = torch.load(file)

                nx_graph = nx.DiGraph()
                nx_graph.add_nodes_from(range(pyg_graph.num_nodes))
                nx_graph.role_embeddings = {}
                task = getattr(pyg_graph, "question", "")

                for i, node_data in enumerate(pyg_graph.x):
                    role = node_data.get("role", id_to_role[0])
                    if role not in role_to_id:
                        role = id_to_role[0]
                    embedding = self.precomputed_embeddings.get(
                        role, torch.zeros_like(next(iter(self.precomputed_embeddings.values())))
                    )
                    nx_graph.nodes[i]["role"] = role
                    nx_graph.nodes[i]["role_id"] = role_to_id[role]
                    nx_graph.nodes[i]["label"] = role_to_id[role]
                    nx_graph.nodes[i]["model"] = node_data.get("model")
                    nx_graph.role_embeddings[i] = embedding

                if hasattr(pyg_graph, "edge_index"):
                    edge_index = pyg_graph.edge_index.numpy()
                    for j in range(edge_index.shape[1]):
                        nx_graph.add_edge(int(edge_index[0, j]), int(edge_index[1, j]), label=0)

                if not nx.is_directed_acyclic_graph(nx_graph):
                    for cycle in list(nx.simple_cycles(nx_graph)):
                        if len(cycle) > 1:
                            nx_graph.remove_edge(cycle[-2], cycle[-1])

                nx_graph.graph["mode"] = getattr(pyg_graph, "mode", "Unknown")
                nx_graph.graph["is_correct"] = getattr(pyg_graph, "is_correct", False)
                nx_graph.graph["agent_nums"] = getattr(pyg_graph, "agent_nums", 1)
                nx_graph.graph["task_embedding"] = get_sentence_embedding(task)
                nx_graph.graph["is_dag"] = True
                nx_graphs.append(nx_graph)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        dag_count = sum(1 for g in nx_graphs if nx.is_directed_acyclic_graph(g))
        print(f"DAG check: {dag_count}/{len(nx_graphs)} graphs are DAGs")
        return nx_graphs

    def __getitem__(self, index):
        return self.graph_list[index]

    def __len__(self):
        return len(self.graph_list)
