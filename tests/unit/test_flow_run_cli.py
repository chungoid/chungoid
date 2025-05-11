import textwrap
from pathlib import Path

from typer.testing import CliRunner

from importlib import util

# Dynamically load CLI module from dev scripts
_cli_path = Path(__file__).resolve().parents[3] / "chungoid-core" / "dev" / "scripts" / "flow_run.py"
spec = util.spec_from_file_location("flow_run_cli", _cli_path)
cli_mod = util.module_from_spec(spec)
spec.loader.exec_module(cli_mod)  # type: ignore

runner = CliRunner()

def _sample_yaml(tmp_path: Path) -> Path:
    yaml_text = textwrap.dedent(
        """
        start_stage: one
        stages:
          one:
            agent_id: a1
          
        """
    )
    p = tmp_path / "flow.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return p

def _branching_yaml(tmp_path: Path) -> Path:
    yaml_text = textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: a1
            next:
              condition: "input == 'force_true_path'" # Condition expects 'input' from context
              "true": s2 # Quoted key
              "false": s_end_false_branch # Quoted key, schema requires a string here
          s2:
            agent_id: a2
          s_end_false_branch: # Dummy stage for schema compliance
            agent_id: a_dummy_end
        """
    )
    p = tmp_path / "branching.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return p

def _input_branching_yaml(tmp_path: Path) -> Path:
    yaml_text = textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: a1
            next:
              condition: "input == 'foo'" # Note: string values in conditions need quotes
              "true": s2 # Quoted key
              "false": s3 # Quoted key
          s2:
            agent_id: a2
          s3:
            agent_id: a3
        """
    )
    p = tmp_path / "input_branching.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return p

def test_cli_sync(tmp_path: Path):
    flow_file = _sample_yaml(tmp_path)
    result = runner.invoke(cli_mod.app, [str(flow_file)])
    assert result.exit_code == 0
    assert "one" in result.output 

def test_cli_branching(tmp_path: Path):
    flow_file = _branching_yaml(tmp_path)
    result = runner.invoke(cli_mod.app, [str(flow_file), "--input", "force_true_path"])
    assert result.exit_code == 0
    assert "s1" in result.output and "s2" in result.output 

def test_cli_input_branching(tmp_path: Path):
    flow_file = _input_branching_yaml(tmp_path)
    # Should go s1 -> s2 if input==foo, else s1 -> s3
    result = runner.invoke(cli_mod.app, [str(flow_file), "--input", "foo"])
    assert result.exit_code == 0
    assert "s1" in result.output and "s2" in result.output
    result2 = runner.invoke(cli_mod.app, [str(flow_file), "--input", "bar"])
    assert result2.exit_code == 0
    assert "s1" in result2.output and "s3" in result2.output 