# Auto-generated stub
# tests/test_energy.py

from quantumflow_ai.modules.q_energy.qbm_scheduler import qbm_schedule
from quantumflow_ai.modules.q_energy.classical_scheduler import classical_schedule
from quantumflow_ai.modules.q_energy.scheduler_utils import normalize_energy_profile
from quantumflow_ai.modules.q_energy.meta_scheduler import MetaScheduler

def mock_job_graph():
    return {
        "jobs": {
            "A": [], 
            "B": ["A"], 
            "C": ["A"]
        }
    }

def mock_energy_profile():
    return {"A": 10.0, "B": 7.5, "C": 5.0}

def test_qbm_scheduler():
    graph = mock_job_graph()
    profile = normalize_energy_profile(mock_energy_profile())
    result = qbm_schedule(graph, profile)
    assert all(job in result for job in graph["jobs"])

def test_classical_scheduler():
    graph = mock_job_graph()
    profile = normalize_energy_profile(mock_energy_profile())
    result = classical_schedule(graph, profile)
    assert all(job in result for job in graph["jobs"])


def test_meta_scheduler_untrained():
    meta = MetaScheduler()
    # Ensure model is in an untrained state
    from sklearn.ensemble import RandomForestClassifier
    meta.model = RandomForestClassifier()
    strategy = meta.recommend([1.0, 2.0, 3.0])
    assert isinstance(strategy, str)


def test_meta_scheduler_feature_padding(tmp_path):
    meta = MetaScheduler()
    meta.model_path = tmp_path / "model.pkl"
    X = [[float(i) for i in range(8)] for _ in range(4)]
    y = ["qbm", "classical", "qaoa", "qbm"]
    meta.train(X, y)
    strategy = meta.recommend([1.0, 2.0, 3.0])
    assert isinstance(strategy, str)
