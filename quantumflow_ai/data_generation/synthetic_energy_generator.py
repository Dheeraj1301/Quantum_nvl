# synthetic_energy_generator.py

import random
import json
import argparse
import os


def generate_synthetic_energy_profiles(profile_name: str, num_graphs=100, num_jobs=8, output_dir="notebooks/profiles"):
    os.makedirs(output_dir, exist_ok=True)
    data = []

    for _ in range(num_graphs):
        job_graph = {"jobs": {}}
        for i in range(num_jobs):
            job_graph["jobs"][f"J{i}"] = [f"J{j}" for j in range(i) if random.random() < 0.4]

        energy_profile = {f"J{i}": round(random.uniform(1.0, 10.0), 2) for i in range(num_jobs)}
        schedule = {f"J{i}": random.randint(0, 10) for i in range(num_jobs)}
        features = [energy_profile[j] * schedule[j] for j in schedule]
        total_energy = sum(features)

        data.append({
            "graph": job_graph,
            "energy_profile": energy_profile,
            "schedule": schedule,
            "features": features,
            "cost": total_energy
        })

    path = os.path.join(output_dir, f"{profile_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved synthetic data to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="Profile name (e.g., A100, H100)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output_dir", default="notebooks/profiles")
    args = parser.parse_args()

    generate_synthetic_energy_profiles(args.profile.lower(), args.samples, output_dir=args.output_dir)
