import random
import pytest
pytest.importorskip("pydantic_settings")
from quantumflow_ai.core.config import set_global_seed
from quantumflow_ai.core.qae_preprocessor import normalize_payload
from quantumflow_ai.core.qadms_selector import select_module
from quantumflow_ai.core.failure_predictor import predict_retry
from quantumflow_ai.core.pipeline_manager import PipelineManager
from quantumflow_ai.core.workflow_optimizer import optimize_order
from quantumflow_ai.core.meta_rl_controller import controller as rl_controller
from quantumflow_ai.core.cross_module_attention import CrossModuleAttention


def test_set_global_seed_reproducible():
    set_global_seed(42)
    val1 = random.random()
    set_global_seed(42)
    val2 = random.random()
    assert val1 == val2


def test_normalize_payload():
    data = {"arr": [1, 2, 3], "num": 5}
    norm = normalize_payload(data)
    assert set(norm) == set(data)


def test_select_module_basic():
    mods = {"foo_quantum": object(), "foo_classical": object()}
    name = select_module("foo", {"latency": 1.0, "queue_size": 1.0}, mods)
    assert name in mods


def test_predict_retry_range():
    score = predict_retry([0.0, 0.0, 0.0])
    assert 0.0 <= score <= 1.0


def test_pipeline_manager_register_and_run():
    pm = PipelineManager()
    pm.register("echo", lambda x: x)
    result = pm.run("echo", 5)
    assert result == 5


def test_pipeline_manager_run_pipeline():
    pm = PipelineManager()
    pm.register("m1", lambda x: {"v": x})
    modules = [{"name": "m1"}]
    res = pm.run_pipeline(modules, {"a": 1})
    assert "m1" in res


def test_optimize_order_topological():
    modules = [
        {"name": "a"},
        {"name": "b", "depends_on": ["a"]},
    ]
    ordered = optimize_order(modules)
    assert ordered[0]["name"] == "a"


def test_meta_rl_controller_toggle():
    rl_controller.enabled = True
    override = rl_controller.recommend_module("x_quantum", {})
    rl_controller.record_outcome("x_quantum", {}, 1.0)
    assert override in {None, "x_classical"}


def test_cross_module_attention_noop():
    att = CrossModuleAttention()
    att.enabled = True
    att.log_sequence(["a", "b"], [1.0, -1.0])
    assert True

