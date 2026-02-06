# Hook Configuration Explanation

## PostToolUse (Write|Edit) - Command Hook
**When it fires:** After Claude writes or edits a file.  
**What it does:** Runs `scripts/check_python.sh`, which reads the tool input JSON from stdin, extracts `tool_input.file_path`, and if the file is a `.py` file it:
- runs `ruff check --fix`
- runs `ruff format`
- runs `python -m py_compile`
This enforces linting, formatting, and syntax validation on every code change.

## PostToolUse (Write|Edit) - Prompt Hook
**When it fires:** After Claude writes or edits a file.  
**What it does:** Prompts Claude to decide whether a pytest test file should be created or updated for the changed Python module, and to place it under a `tests/` directory.

## PreToolUse (Bash) - Command Hook
**When it fires:** Before Claude executes any Bash command.  
**What it does:** Runs `scripts/block_force_push.sh`, which reads the tool input JSON from stdin, extracts `tool_input.command`, and blocks any `git push` command that uses `-f` or `--force` to prevent destructive force pushes.
