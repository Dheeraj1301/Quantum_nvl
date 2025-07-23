from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from quantumflow_ai.modules.q_energy.qbm_scheduler import qbm_schedule
from quantumflow_ai.modules.q_energy.classical_scheduler import classical_schedule
from quantumflow_ai.modules.q_energy.scheduler_utils import normalize_energy_profile
from quantumflow_ai.modules.q_energy.ml_scheduler_predictor import MLEnergyPredictor
from quantumflow_ai.modules.q_energy.qaoa_dependency_scheduler import qaoa_schedule
from quantumflow_ai.modules.q_energy.hybrid_scheduler import hybrid_schedule
from quantumflow_ai.modules.q_energy.meta_scheduler import MetaScheduler
from quantumflow_ai.modules.q_energy.gnn_predictor import predict_energy_with_gnn

router = APIRouter()

class EnergyInput(BaseModel):
    job_graph: dict
    energy_profile: dict
    use_quantum: bool = True
    use_ml: bool = False
    use_qaoa: bool = False
    use_hybrid: bool = True
    use_meta: bool = False
    use_gnn: bool = False
    max_energy_limit: int | None = None
    quantum_iterations: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None

@router.post("/q-energy/schedule")
def schedule_energy(input: EnergyInput):
    try:
        ep = normalize_energy_profile(input.energy_profile)

        # 🔮 QAOA path
        if input.use_qaoa:
            qaoa_result = qaoa_schedule(input.job_graph)
            return {"qaoa_schedule": qaoa_result}

        # 🧠 GNN-only prediction mode
        if input.use_gnn:
            predicted_cost = predict_energy_with_gnn(input.job_graph, ep)
            return {"gnn_predicted_cost": predicted_cost}

        # 🧠 Meta Strategy prediction
        if input.use_meta:
            schedule = classical_schedule(input.job_graph, ep)
            feat = MLEnergyPredictor().schedule_to_features(schedule, ep)
            strategy = MetaScheduler().recommend(feat)
            return {"meta_strategy": strategy}

        # ⚙️ Hybrid Optimization
        if input.use_hybrid:
            result = hybrid_schedule(input.job_graph, ep)
            return result

        # 🧮 Manual path: quantum/classical + optional ML
        raw_schedule = qbm_schedule(input.job_graph, ep) if input.use_quantum else classical_schedule(input.job_graph, ep)
        final_schedule = MLEnergyPredictor().suggest_reschedule(raw_schedule, ep) if input.use_ml else raw_schedule

        return {
            "raw_schedule": raw_schedule,
            "final_schedule": final_schedule
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
