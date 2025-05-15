import asyncio
import os
import shutil
import tempfile
from pathlib import Path
import pytest

from chungoid.schemas.agent_code_integration import CodeIntegrationInput, CodeIntegrationOutput
from chungoid.runtime.agents.core_code_integration_agent import CoreCodeIntegrationAgentV1

@pytest.fixture
def agent() -> CoreCodeIntegrationAgentV1:
    return CoreCodeIntegrationAgentV1()

@pytest.fixture
def temp_test_dir():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup: remove the temporary directory
    shutil.rmtree(temp_dir)

class TestCoreCodeIntegrationAgentV1:
    async def test_append_to_existing_file(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "test_append.py"
        initial_content = "def hello():\n    print(\"Hello\")\n"
        with open(target_file, "w") as f:
            f.write(initial_content)

        code_to_add = "def world():\n    print(\"World\")"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=code_to_add,
            edit_action="APPEND"
        )

        output = await agent.invoke_async(inputs)

        assert output.status == "SUCCESS"
        assert output.modified_file_path == str(target_file)
        assert output.backup_file_path is None

        with open(target_file, "r") as f:
            content = f.read()
        
        expected_content = initial_content + "\n" + code_to_add + "\n" # Agent adds \n if not present, and code_to_add also gets one if it doesn't have one
        # Let's be more precise based on current agent logic:
        # 1. initial_content ends with \n.
        # 2. agent's append logic: if file not empty and not endswith \n, it adds one. Our initial_content *does* end with \n.
        #    So, no *extra* \n is added by the "ensure a newline before appending" part.
        # 3. The code_to_add itself will be written.
        # Let's re-check agent's APPEND logic:
        #   content_to_add = inputs.code_to_integrate
        #   if target_path.stat().st_size > 0:
        #       with open(target_path, 'r', encoding='utf-8') as f_read:
        #           if not f_read.read().endswith('\n'): # Our initial_content *does* end with \n
        #               content_to_add = "\n" + content_to_add # This won't run
        #   with open(target_path, "a", encoding="utf-8") as f:
        #       f.write(content_to_add) # Writes code_to_add
        # So, if code_to_add doesn't end with \n, the final file might not have it on the very last line.
        # The test should reflect that or the agent should ensure the appended block ends with \n.
        # For APPEND, it doesn't explicitly add a newline to the appended block itself.
        # Let's assume code_to_add might or might not have a newline.
        
        expected_content = initial_content + code_to_add # Direct append
        assert content == expected_content

    async def test_append_to_existing_file_needs_leading_newline(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "test_append_nl.py"
        initial_content = "def hello():\n    print(\"Hello\")" # No trailing newline
        with open(target_file, "w") as f:
            f.write(initial_content)

        code_to_add = "def world():\n    print(\"World\")"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=code_to_add,
            edit_action="APPEND"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        expected_content = initial_content + "\n" + code_to_add
        assert content == expected_content

    async def test_append_to_non_existent_file_fails(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "ghost_file.py"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate="print(\"Test\")",
            edit_action="APPEND"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "FAILURE"
        assert not target_file.exists()

    async def test_append_with_backup(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "test_backup_append.py"
        initial_content = "original_content"
        with open(target_file, "w") as f:
            f.write(initial_content)

        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate="new_content",
            edit_action="APPEND",
            backup_original=True
        )
        output = await agent.invoke_async(inputs)

        assert output.status == "SUCCESS"
        assert output.modified_file_path == str(target_file)
        backup_file = target_file.with_suffix(target_file.suffix + ".bak")
        assert output.backup_file_path == str(backup_file)
        assert backup_file.exists()
        
        with open(backup_file, "r") as f:
            backup_content = f.read()
        assert backup_content == initial_content

        with open(target_file, "r") as f:
            current_content = f.read()
        # Based on agent logic: initial_content (no \n) + \n + "new_content"
        assert current_content == initial_content + "\n" + "new_content"

    # Tests for CREATE_OR_APPEND
    async def test_create_or_append_creates_new_file(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "new_file_for_create.py"
        code_to_add = "# New file content"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=code_to_add,
            edit_action="CREATE_OR_APPEND"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        assert target_file.exists()
        with open(target_file, "r") as f:
            content = f.read()
        assert content == code_to_add # CREATE_OR_APPEND for new file writes directly.

    async def test_create_or_append_appends_to_existing_file(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "existing_for_create_append.py"
        initial_content = "line1\nline2" # Ends without newline
        with open(target_file, "w") as f:
            f.write(initial_content)
        
        code_to_add = "line3"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=code_to_add,
            edit_action="CREATE_OR_APPEND"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        # Expected: initial_content + \n + code_to_add
        assert content == initial_content + "\n" + code_to_add

    async def test_create_or_append_to_directory_fails(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_dir = temp_test_dir / "my_dir"
        target_dir.mkdir()
        
        inputs = CodeIntegrationInput(
            target_file_path=str(target_dir), # Targeting the directory itself
            code_to_integrate="some code",
            edit_action="CREATE_OR_APPEND"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "FAILURE"
        assert "is a directory" in output.message

    # TODO: Add tests for ADD_PYTHON_IMPORTS
    # TODO: Add tests for ADD_TO_CLICK_GROUP
    # TODO: Test cases for empty file inputs for append/create_or_append
    # TODO: Test cases for code_to_integrate having leading/trailing newlines already

    # --- Tests for ADD_PYTHON_IMPORTS ---

    async def test_add_imports_to_empty_file(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "empty_for_imports.py"
        target_file.touch() # Create empty file

        imports_to_add = "import os\nfrom pathlib import Path"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=imports_to_add,
            edit_action="ADD_PYTHON_IMPORTS"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        # Agent adds \n to each import if not present, and if no existing imports and file becomes non-empty,
        # it might add an extra \n at the end of imports_to_add list.
        # Current logic: if last_import_line_index == -1 and lines (original) is empty, no extra \n added by imports_to_add.append("\n")
        # Each import in imports_to_add gets a \n.
        expected_content = "import os\nfrom pathlib import Path\n"
        assert content == expected_content

    async def test_add_imports_to_file_with_no_imports_but_code(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "code_no_imports.py"
        initial_content = "def my_function():\n    print(\"Hello\")"
        with open(target_file, "w") as f:
            f.write(initial_content)

        imports_to_add_str = "import sys"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=imports_to_add_str,
            edit_action="ADD_PYTHON_IMPORTS"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        # Expected: import sys\n\n# blank line #\ndef my_function()...
        # Agent logic: if last_import_line_index == -1 and lines and lines[0].strip(): imports_to_add.append("\n")
        expected_content = "import sys\n\n" + initial_content
        assert content == expected_content

    async def test_add_imports_to_file_with_existing_imports(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "existing_imports.py"
        initial_content = "import os\n\ndef my_function():\n    print(\"Hello\")"
        with open(target_file, "w") as f:
            f.write(initial_content)

        imports_to_add_str = "from pathlib import Path\nimport sys"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=imports_to_add_str,
            edit_action="ADD_PYTHON_IMPORTS"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        # Expected: import os\nfrom pathlib import Path\nimport sys\n\ndef my_function()...
        # Agent logic for newline after new imports:
        # if next line is not empty and not a comment, and last new import has content, insert \n
        # initial_content has "import os\n" then "\n" then "def my_function..."
        # Last import is 'import os' at index 0.
        # New imports 'from pathlib import Path\n', 'import sys\n' are inserted after line 0.
        # lines becomes: ['import os\n', 'from pathlib import Path\n', 'import sys\n', '\n', 'def my_function():\n', '    print("Hello")']
        # new_last_import_idx is 0 + 2 = 2 (points to 'import sys\n')
        # lines[new_last_import_idx + 1] is '\n'.strip() which is empty. So, no extra newline inserted.
        expected_content = "import os\nfrom pathlib import Path\nimport sys\n\ndef my_function():\n    print(\"Hello\")"
        assert content == expected_content

    async def test_add_duplicate_imports_are_skipped(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "duplicate_imports.py"
        initial_content = "import os\nfrom pathlib import Path\n\ndef main():\n    pass"
        with open(target_file, "w") as f:
            f.write(initial_content)

        imports_to_add_str = "import os\nimport sys\nfrom pathlib import Path" # os and Path are duplicates
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=imports_to_add_str,
            edit_action="ADD_PYTHON_IMPORTS"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        # Expected: only 'import sys' is added.
        # Original: import os\nfrom pathlib import Path\n\ndef main():
        # Last import 'from pathlib import Path' is at index 1.
        # 'import sys\n' is added after index 1.
        # lines: ['import os\n', 'from pathlib import Path\n', 'import sys\n', '\n', 'def main():\n', '    pass']
        # new_last_import_idx = 1 + 1 = 2 (points to 'import sys\n')
        # lines[new_last_import_idx+1] is '\n'.strip() which is empty. So no extra blank line.
        expected_content = "import os\nfrom pathlib import Path\nimport sys\n\ndef main():\n    pass"
        assert content == expected_content

    async def test_add_imports_all_duplicates(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "all_duplicate_imports.py"
        initial_content = "import os\nfrom pathlib import Path"
        with open(target_file, "w") as f:
            f.write(initial_content)

        imports_to_add_str = "import os\nfrom pathlib import Path"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=imports_to_add_str,
            edit_action="ADD_PYTHON_IMPORTS"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS" # Agent returns SUCCESS if no new imports needed
        assert "No new imports needed" in output.message
        with open(target_file, "r") as f:
            content = f.read()
        assert content == initial_content # No change

    async def test_add_imports_to_file_with_imports_no_trailing_code_newline_logic(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "imports_no_code.py"
        initial_content = "import os\n" # Ends with a newline
        with open(target_file, "w") as f:
            f.write(initial_content)

        imports_to_add_str = "import sys"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=imports_to_add_str,
            edit_action="ADD_PYTHON_IMPORTS"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        # Last import 'import os' at index 0.
        # New import 'import sys\n' added after it.
        # lines: ['import os\n', 'import sys\n']
        # new_last_import_idx = 0 + 1 = 1 ('import sys\n')
        # (new_last_import_idx + 1) which is 2, is NOT < len(lines) which is 2.
        # So, no blank line insertion logic is triggered.
        expected_content = "import os\nimport sys\n"
        assert content == expected_content

    async def test_add_imports_non_existent_file_fails(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "ghost_imports.py"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate="import os",
            edit_action="ADD_PYTHON_IMPORTS"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "FAILURE"
        assert "does not exist or is not a file" in output.message
        assert not target_file.exists()

    async def test_add_python_imports_creates_parent_dirs_but_file_must_exist(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        deep_dir = temp_test_dir / "deep" / "down"
        target_file = deep_dir / "deep_imports.py"
        # Parent dirs do not exist yet

        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate="import os",
            edit_action="ADD_PYTHON_IMPORTS"
        )
        # This should fail because ADD_PYTHON_IMPORTS requires the file to exist
        # even if it creates parent directories.
        output = await agent.invoke_async(inputs)
        assert output.status == "FAILURE"
        assert "does not exist or is not a file" in output.message
        assert deep_dir.exists() # Parent dirs should be created by the generic logic
        assert not target_file.exists()

        # Now, create the file and try again
        target_file.touch()
        output_retry = await agent.invoke_async(inputs)
        assert output_retry.status == "SUCCESS"
        assert target_file.exists()
        with open(target_file, "r") as f:
            content = f.read()
        assert content == "import os\n"

    # --- Tests for ADD_TO_CLICK_GROUP ---

    async def test_add_to_click_group_empty_file(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "empty_for_click.py"
        target_file.touch()

        code_to_add = "@click.group()\ndef cli():\n    pass\n\n@cli.command()\ndef my_cmd():\n    print(\"Hello Click\")"
        
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=code_to_add,
            edit_action="ADD_TO_CLICK_GROUP",
            integration_point_hint="cli", # Not used by V1 append logic, but good to pass
            click_command_name="my_cmd"  # Not used by V1 append logic
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        # V1 appends. For an empty file, prepended_newlines = "". Block ensures it ends with \n.
        expected_content = code_to_add + "\n" if not code_to_add.endswith("\n") else code_to_add
        assert content == expected_content

    async def test_add_to_click_group_existing_content(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "existing_code_for_click.py"
        initial_content = "import click\n\n# Some other code\ndef helper():\n    return 42\n"
        with open(target_file, "w") as f:
            f.write(initial_content)

        code_to_add = "@click.group()\ndef cli():\n    pass\n\n@cli.command()\ndef new_command():\n    print(\"New Click command\")"

        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=code_to_add,
            edit_action="ADD_TO_CLICK_GROUP"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()
        
        # Agent logic for ADD_TO_CLICK_GROUP newline prepending:
        # initial_content ends with \n. prepended_newlines will be "\n" (needs one more to make two total)
        # final_code_to_add ensures the block itself ends with a newline.
        expected_prepended_newlines = "\n"
        expected_code_block = code_to_add
        if not expected_code_block.endswith("\n"):
            expected_code_block += "\n"
        
        expected_content = initial_content + expected_prepended_newlines + expected_code_block
        assert content == expected_content

    async def test_add_to_click_group_existing_content_no_final_newline(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "click_no_final_nl.py"
        initial_content = "import click\ndef old_func(): pass" # No final newline
        with open(target_file, "w") as f:
            f.write(initial_content)

        code_to_add = "@click.command()\ndef new_cmd(): pass"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=code_to_add,
            edit_action="ADD_TO_CLICK_GROUP"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        with open(target_file, "r") as f:
            content = f.read()

        # initial_content does not end with \n. prepended_newlines = "\n" + "\n\n" = "\n\n\n" -> this is wrong, should be "\n\n\n"
        # Agent logic: if not content.endswith('\n'): prepended_newlines = "\n" + prepended_newlines. default is \n\n. So \n\n\n
        # Let's re-verify agent: default prepended_newlines = "\n\n"
        # if not content.endswith('\n'): prepended_newlines = "\n" + prepended_newlines # -> "\n\n\n"
        # elif content.endswith('\n\n'): prepended_newlines = "" # Has two, need zero more
        # elif content.endswith('\n'): prepended_newlines = "\n" # Has one, need one more
        # In this test case, initial_content does not end with \n. So prepended_newlines = "\n\n\n"
        expected_prepended_newlines = "\n\n\n"
        expected_code_block = code_to_add
        if not expected_code_block.endswith("\n"):
            expected_code_block += "\n"
        expected_content = initial_content + expected_prepended_newlines + expected_code_block
        assert content == expected_content

    async def test_add_to_click_group_with_backup(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "click_with_backup.py"
        initial_content = "import click"
        with open(target_file, "w") as f:
            f.write(initial_content)
        
        code_to_add = "@click.command()\ndef my_cmd(): pass"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate=code_to_add,
            edit_action="ADD_TO_CLICK_GROUP",
            backup_original=True
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "SUCCESS"
        assert output.backup_file_path is not None
        backup_file = Path(output.backup_file_path)
        assert backup_file.exists()
        with open(backup_file, "r") as f:
            assert f.read() == initial_content

    async def test_add_to_click_group_non_existent_file_fails(self, agent: CoreCodeIntegrationAgentV1, temp_test_dir: Path):
        target_file = temp_test_dir / "ghost_click.py"
        inputs = CodeIntegrationInput(
            target_file_path=str(target_file),
            code_to_integrate="@click.command()\ndef test(): pass",
            edit_action="ADD_TO_CLICK_GROUP"
        )
        output = await agent.invoke_async(inputs)
        assert output.status == "FAILURE"
        assert "does not exist or is not a file" in output.message
        assert not target_file.exists()