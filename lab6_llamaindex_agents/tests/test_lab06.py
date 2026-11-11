import os
import json
from pathlib import Path
import pytest
import pandas as pd

LAB6_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB6_DIR / "data"
OUTPUTS_DIR = LAB6_DIR / "outputs"
TRACES_DIR = OUTPUTS_DIR / "traces"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

def test_data_files_exist():
    registry_file = DATA_DIR / "models_registry.json"
    runbook_file = DATA_DIR / "mlops_runbook.txt"
    assert registry_file.exists(), f"Missing {registry_file}"
    assert runbook_file.exists(), f"Missing {runbook_file}"

def test_models_registry_schema():
    registry_file = DATA_DIR / "models_registry.json"
    with open(registry_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "models" in data, "Registry JSON must have 'models' key"
    models = data["models"]
    assert len(models) >= 5, f"Expected >= 5 models, got {len(models)}"
    
    required_keys = {"model_id", "framework", "version", "status", "p99_latency_ms", "slo_latency_ms", "slo_availability_pct"}
    for m in models:
        assert required_keys.issubset(m.keys()), f"Model {m.get('model_id')} missing keys: {required_keys - set(m.keys())}"

def test_runbook_content_sections():
    runbook_file = DATA_DIR / "mlops_runbook.txt"
    with open(runbook_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "## Section 1: Latency Violation Protocols" in content
    assert "## Section 2: Error Rate and Availability Degradation" in content
    assert "churn_predictor_v3" in content
    assert "llm_router_v1" in content

def test_traces_generated():
    traces = list(TRACES_DIR.glob("*.json"))
    assert len(traces) >= 9, f"Expected at least 9 scenario traces, got {len(traces)}"
    
    for tr in traces:
        with open(tr, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "scenario_id" in data
        assert "category" in data
        assert "final_response" in data
        assert len(data["final_response"]) > 0

def test_evaluation_table():
    csv_file = TABLES_DIR / "agent_evaluation_results.csv"
    assert csv_file.exists(), f"Missing {csv_file}"
    df = pd.read_csv(csv_file)
    assert len(df) >= 9
    assert "scenario_id" in df.columns
    assert "tool_count" in df.columns
    assert "task_success" in df.columns
    assert df["task_success"].all(), "All benchmark tasks should be successfully completed"

def test_figures_exist():
    fig1 = FIGURES_DIR / "scenario_latency_breakdown.png"
    fig2 = FIGURES_DIR / "tool_invocation_distribution.png"
    assert fig1.exists() and fig1.stat().st_size > 1000
    assert fig2.exists() and fig2.stat().st_size > 1000
