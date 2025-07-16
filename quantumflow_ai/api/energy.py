from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from quantumflow_ai.modules.q_energy.qbm_scheduler import qbm_schedule
from quantumflow_ai.modules.q_energy.classical_scheduler import classical_schedule
from quantumflow_ai.modules.q_energy.scheduler_utils import normalize_energy_profile
from quantumflow_ai.modules.q_energy.ml_scheduler_predictor import MLEnergyPredictor
from quantumflow_ai.modules.q_energy.qaoa_dependency_scheduler import qaoa_schedule
from quantumflow_ai.modules.q_energy.hybrid_scheduler import hybrid_schedule

router = APIRouter()

class EnergyInput(BaseModel):
    job_graph: dict
    energy_profile: dict
    use_quantum: bool = True
    use_ml: bool = False
    use_qaoa: bool = False
    use_hybrid: bool = True

@router.post("/q-energy/schedule")
def schedule_energy(input: EnergyInput):
    try:
        ep = normalize_energy_profile(input.energy_profile)

        if input.use_qaoa:
            qaoa_result = qaoa_schedule(input.job_graph)
            return {"qaoa_schedule": qaoa_result}

        if input.use_hybrid:
            result = hybrid_schedule(input.job_graph, ep)
            return result

        raw_schedule = qbm_schedule(input.job_graph, ep) if input.use_quantum else classical_schedule(input.job_graph, ep)
        final_schedule = MLEnergyPredictor().suggest_reschedule(raw_schedule, ep) if input.use_ml else raw_schedule

        return {
            "raw_schedule": raw_schedule,
            "final_schedule": final_schedule
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
