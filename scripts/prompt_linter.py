import re
import sys
from pathlib import Path
import yaml

# Configs
MAX_TOKENS_EST = 2000  # rough estimate = words
PLACEHOLDER_PATTERN = re.compile(r"{{\s*(\w+)\s*}}")
# TODO: Review and expand this list as needed for chungoid-core prompts
ALLOWED_PLACEHOLDERS = {
    "reflections_context",
    "project_context",
    "agent_name",
    "artifacts_context",
    # Add other common placeholders for chungoid-core here
}


def _estimate_tokens(text: str) -> int:  # noqa: D401
    # very rough: word count as proxy
    return len(text.strip().split())


def lint_file(path: Path) -> list[str]:  # noqa: D401
    errors: list[str] = []
    raw = path.read_text(encoding="utf-8")
    # Check unmatched braces
    open_cnt = raw.count("{{")
    close_cnt = raw.count("}}");
    if open_cnt != close_cnt:
        errors.append(f"Unbalanced '{{' '}}' pairs (found {open_cnt} '{{', {close_cnt} '}}').")
    # Placeholder validation
    for m in PLACEHOLDER_PATTERN.finditer(raw):
        name = m.group(1)
        if name not in ALLOWED_PLACEHOLDERS:
            errors.append(f"Unknown placeholder '{{{{ {name} }}}}'")
    # Token estimate on system_prompt & prompt_details keys
    try:
        doc = yaml.safe_load(raw)
    except Exception as exc:
        errors.append(f"YAML parse error: {exc}")
        return errors
    for key in ("system_prompt", "prompt_details", "user_prompt"):
        if key in doc and isinstance(doc[key], str):
            tok = _estimate_tokens(doc[key])
            if tok > MAX_TOKENS_EST:
                errors.append(f"{key} exceeds {MAX_TOKENS_EST} token estimate ({tok}).")
    return errors


def main():  # noqa: D401
    # Assumes script is in chungoid-core/scripts/
    # Targets server_prompts directory within chungoid-core
    script_dir = Path(__file__).parent
    core_project_root = script_dir.parent 
    targets_dir = core_project_root / "server_prompts"
    
    if not targets_dir.exists() or not targets_dir.is_dir():
        print(f"Error: Prompts directory not found at {targets_dir}", file=sys.stderr)
        sys.exit(1)

    targets = list(targets_dir.rglob("*.yaml"))
    if not targets:
        print(f"No YAML files found in {targets_dir}", file=sys.stderr)
        # Exiting with 0 as it's not a linting failure, just no files to lint.
        # This might be an error condition in some CI setups though.
        print("✓ Prompt linter passed (no files to lint).")
        sys.exit(0)
        
    failed = False
    for p in targets:
        errs = lint_file(p)
        if errs:
            failed = True
            print(f"\n[prompt-linter] {p}:")
            for e in errs:
                print("  -", e)
    if failed:
        print("\n✖ Prompt linter failed.")
        sys.exit(1)
    else:
        print("✓ Prompt linter passed.")


if __name__ == "__main__":
    main() 