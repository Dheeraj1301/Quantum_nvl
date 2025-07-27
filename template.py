# template.py
import os

PROJECT_STRUCTURE = {
    "frontend": {
        "dashboards": ["__init__.py", "routing.py", "energy.py", "compression.py", "home.py"],
        "utils": ["__init__.py", "layout.py"],
        "public": []
    },
    "modules": {
        "q_routing": ["__init__.py", "qaoa_router.py", "classical_router.py", "simulator.py"],
        "q_energy": [
            "__init__.py",
            "qbm_scheduler.py",
            "classical_scheduler.py",
            "hybrid_scheduler.py",
            "qaoa_dependency_scheduler.py",
            "meta_scheduler.py",
            "ml_scheduler_predictor.py",
            "gnn_predictor.py",
            "scheduler_utils.py"
        ],
        "q_compression": [
            "__init__.py",
            "q_autoencoder.py",
            "classical_compressor.py",
            "simulator.py",
            "denoiser.py",
            "vqc_classifier.py"
        ],
        "q_decompression": [
            "__init__.py",
            "qft_decoder.py",
            "hhl_solver.py",
            "qsvt_solver.py",
            "ae_refiner.py",
            "lstm_enhancer.py"
        ],
        "q_hpo": [
            "__init__.py",
            "vqc_optimizer.py",
            "classical_optimizer.py",
            "hybrid_optimizer.py",
            "meta_lstm_predictor.py",
            "quantum_kernel_decoder.py",
            "vqc_regressor.py",
            "context_embedder.py",
            "search_space.py",
            "gumbel_sampler.py",
            "warm_start.py"
        ],
        "q_nvlinkopt": [
            "__init__.py",
            "nvlink_graph_optimizer.py",
            "nvlink_simulator.py",
            "qgnn_hybrid_optimizer.py",
            "quantum_graph_kernel.py",
            "vqaoa_balancer.py",
            "quantum_routing_rl.py",
            "topology_qclassifier.py",
            "classical_nvlink_planner.py"
        ],
        "q_attention": [
            "__init__.py",
            "qka_kernel_attention.py",
            "vqc_reweight_attention.py",
            "qsvd_head_pruner.py",
            "qpe_position_encoder.py",
            "contrastive_trainer.py",
            "qaoa_sparse_attention.py",
            "hybrid_transformer_layer.py",
            "classical_attention.py",
            "utils.py"
        ],
        "optional_integration": ["__init__.py", "nemo_adapter.py", "tensorrt_hook.py"]
    },
    "optimizers": ["__init__.py", "gen_routing.py", "gen_energy.py", "gen_compression.py"],
    "core": [
        "__init__.py",
        "quantum_backend.py",
        "classical_fallback.py",
        "pipeline_manager.py",
        "config.py",
        "logger.py",
        "circuit_visualizer.py",
        "failure_predictor.py",
        "workflow_optimizer.py",
        "meta_rl_controller.py",
        "qadms_selector.py",
        "cross_module_attention.py",
        "qae_preprocessor.py"
    ],
    "api": [
        "__init__.py",
        "main.py",
        "q_router.py",
        "energy.py",
        "decompressor.py",
        "hpo.py",
        "compressor.py",
        "nvlinkopt.py",
        "attention.py"
    ],
    "automation": ["n8n_flows.md", "webhook_examples.json"],
    "notebooks": {
        "benchmarks": ["routing_benchmark.ipynb", "energy_benchmark.ipynb"],
        "profiles": []
    },
    "lstm": ["lstm_model.py", "routing_log.py"],
    "data_generation": ["routing_synthetic.py", "synthetic_energy_generator.py"],
    "tests": [
        "test_routing.py",
        "test_energy.py",
        "test_compression.py",
        "test_nvlinkopt.py",
        "test_cli.py"
    ]
}

TOP_LEVEL_FILES = [
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "qml-requirements.txt",
    "README.md",
    ".env"
]

def create_structure(base_path="quantumflow_ai"):
    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"[✔] Created directory: {path}")

    def create_files(path, files):
        for file in files:
            file_path = os.path.join(path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    if file.endswith(".py"):
                        f.write("# Auto-generated stub\n")
                    elif file.endswith(".ipynb"):
                        f.write("")  # Placeholder, can be filled later
                    elif file.endswith(".md"):
                        f.write("# n8n Flows\n")
                    elif file.endswith(".json"):
                        f.write("{}")
                    elif file == ".env":
                        f.write("USE_GPU=True\nQUANTUM_BACKEND=default.qubit\n")
                print(f"[+] Created file: {file_path}")

    ensure_dir(base_path)
    
    for top_folder, content in PROJECT_STRUCTURE.items():
        folder_path = os.path.join(base_path, top_folder)
        if isinstance(content, list):
            ensure_dir(folder_path)
            create_files(folder_path, content)
        elif isinstance(content, dict):
            ensure_dir(folder_path)
            for subfolder, files in content.items():
                subfolder_path = os.path.join(folder_path, subfolder)
                ensure_dir(subfolder_path)
                create_files(subfolder_path, files)

    # Top-level files
    for top_file in TOP_LEVEL_FILES:
        file_path = os.path.join(base_path, top_file)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                if top_file == "requirements.txt":
                    f.write("# Install with pip install -r requirements.txt\n")
                elif top_file == "README.md":
                    f.write("# QuantumFlow_AI\n")
            print(f"[+] Created top-level file: {file_path}")

if __name__ == "__main__":
    create_structure()
