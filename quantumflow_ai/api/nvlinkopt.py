from fastapi import APIRouter, UploadFile, File
import pandas as pd
import networkx as nx
try:
    import torch
except Exception:  # pragma: no cover - optional dependency missing
    torch = None
from quantumflow_ai.modules.q_nvlinkopt.qgnn_hybrid_optimizer import QAOA_GNN_Router
from quantumflow_ai.modules.q_nvlinkopt.quantum_graph_kernel import QuantumGraphEmbedder
from quantumflow_ai.modules.q_nvlinkopt.vqaoa_balancer import VQAOABalancer
from quantumflow_ai.modules.q_nvlinkopt.quantum_routing_rl import QuantumRoutingAgent
from quantumflow_ai.modules.q_nvlinkopt.topology_qclassifier import QuantumTopoClassifier

router = APIRouter()

@router.post("/q-nvlinkopt/run")
async def run_nvlink_optimizers(
    file: UploadFile = File(...),
    use_qgnn: bool = False,
    use_kernel: bool = False,
    use_vqaoa: bool = False,
    use_rl: bool = False,
    use_classifier: bool = False
):
    df = pd.read_csv(file.file)
    G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True)

    output = {}

    if use_qgnn:
        model = QAOA_GNN_Router()
        x, edge_index = model.prepare_input(G)
        routing = model(x, edge_index).detach().numpy()
        output["qgnn_plan"] = routing.tolist()

    if use_kernel:
        graphs = [G.subgraph(list(G.nodes)[i:i+4]) for i in range(0, len(G.nodes), 4)]
        embedder = QuantumGraphEmbedder()
        output["kernel_embeddings"] = embedder.embed_graphs(graphs).tolist()

    if use_vqaoa:
        vqaoa = VQAOABalancer()
        output["vqaoa"] = list(vqaoa.optimize())

    if use_rl:
        agent = QuantumRoutingAgent(state_dim=3, action_dim=3)
        if torch is None:
            raise RuntimeError("PyTorch is required for RL option")
        dummy_state = torch.rand((1, 3))
        action = agent.select_action(dummy_state)
        output["qrl_action"] = int(action)

    if use_classifier:
        features = [
            df["bandwidth"].mean(),
            df["bandwidth"].std(),
            len(G.edges),
            len(G.nodes)
        ]
        topo = QuantumTopoClassifier()
        label, score = topo.classify(features)
        output["topo_class"] = label
        output["class_confidence"] = score

    return output
