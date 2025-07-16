import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_module_help():
    result = subprocess.run(
        [sys.executable, '-m', 'quantumflow_ai', '--help'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'QuantumFlow utilities' in result.stdout


def test_script_help():
    script = REPO_ROOT / 'quantumflow_ai' / '__main__.py'
    result = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert 'QuantumFlow utilities' in result.stdout


def test_alias_for_script():
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'quantumflow_ai',
            'notebooks/train_meta_model.py',
            '--help',
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # Should display the train-meta subcommand help
    assert 'train-meta' in result.stdout
