# hybrid_scheduler.py

from modules.q_energy.qbm_scheduler import qbm_schedule
from modules.q_energy.classical_scheduler import classical_schedule
from modules.q_energy.ml_scheduler_predictor import MLEnergyPredictor
from modules.q_energy.scheduler_utils import normalize_energy_profile

def hybrid_schedule(job_graph, energy_profile):
    normalized_energy = normalize_energy_profile(energy_profile)
    try:
        qbm_result = qbm_schedule(job_graph, normalized_energy)
    except Exception:
        qbm_result = classical_schedule(job_graph, normalized_energy)

    classical_refined = classical_schedule(job_graph, normalized_energy)
    ml_model = MLEnergyPredictor()

    if hasattr(ml_model, "model"):
        ml_result = ml_model.suggest_reschedule(classical_refined, normalized_energy)
        return {
            "qbm_schedule": qbm_result,
            "refined_schedule": classical_refined,
            "final_schedule": ml_result
        }

    return {
        "qbm_schedule": qbm_result,
        "final_schedule": classical_refined
    }
