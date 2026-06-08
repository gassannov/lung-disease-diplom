# Repository Guidelines

## Git Ignore Rules

Ignore git completely in this directory.

- Do not inspect git status, branches, history, diffs, or remotes.
- Do not create branches, commits, tags, stashes, or pull requests.
- Do not use git for workflow, verification, or handoff.
- Treat the working directory contents as the only source of truth.

## Workflow

Develop in this local directory and treat it as the source of truth. Synchronize it to the server with Mutagen and run project commands on the server through SSH, not on the laptop.

- Write code and scripts so they run on the server directly without local wrapper scripts.
- Do not introduce names, variables, or comments like `remote_*` when they refer to the normal runtime on `JH`; describe the server runtime directly.
- Any bash command that touches the project must be executed on the server through `ssh JH ...`.
- Before every server-side command, flush the Mutagen session so the server sees the latest local state.

## Remote Execution Rules

- Default SSH host is `JH` because that alias is configured locally and works non-interactively.
- If `ssh-JH` is added to local SSH config later, set `SSH_HOST=ssh-JH` instead of rewriting commands across the repository.
- The `JH` server has outbound access only to Hugging Face and `pip` package sources. For anything else, download files locally on the laptop and transfer them to the server through the sync workflow or SSH.
- `/home/margasanov` is limited to `20GB`; place large files, datasets, model weights, caches, and heavy artifacts under `/opt/gen-content/margasanov` instead.
- Use only the conda environment at `/opt/gen-content/margasanov/envs/pizding-kartocheck` for all project work on `JH`.
- Use conda as the project package manager and runtime environment manager.
- Do not require `uv` in commands, scripts, or documentation.
- Do not run project commands on `JH` from `base` or any other environment.
- Before the first run, ensure the daemon is available with `mutagen daemon start`.
- Start sync from the repo root with `mutagen project start`.
- Before each command execution, run `mutagen project flush`.
- Check sync health with `mutagen project list` or `mutagen sync list`.
- Use on the server for pip `https://artifactory.s.o3.ru/artifactory/api/pypi/pypi-virtual/simple` only and for HF `https://huggingface.artifactory.s.o3.ru/artifactory/api/huggingfaceml/huggingface-remote` only.

## Build, Test, and Development Commands

Run build, test, lint, and application commands only on `JH` after `mutagen project flush` and inside `/opt/gen-content/margasanov/envs/pizding-kartocheck`.

## Coding Style & Naming Conventions

Write simple, direct code. Do not add defensive checks unless they are required by an actual failure mode. Use 4-space indentation and never leave spaces on empty lines. Avoid unnecessary comments.

Every function and class must include a docstring. Function docstrings must describe purpose, input parameters, and return value. Class docstrings must describe the class and include a short usage example. Follow existing Python naming: `snake_case` for functions and modules, `PascalCase` for classes, and clear handler names such as `search.py` or `errors.py`.

Keep code as simple and readable as possible. Do not add extra functions, checks, abstractions, or helper layers unless they are necessary for the task. Prefer splitting logic into focused files and meaningful blocks instead of growing large files. Keep docstrings short and practical.

## Testing Rules

Do not run tests, write tests, or add test-related work unless the user asks for testing directly.

## Diploma Report Generation

Generate the diploma report as LaTeX sources first: Codex should edit chapter `.tex` files and keep the final PDF as a build artifact. Use Python scripts only for reproducible generated fragments such as tables, metrics, project structure, figures, and appendices, then include them from LaTeX. Build the final PDF through the LaTeX toolchain from `main.tex`; do not generate the PDF directly from model output.

Write diploma text in an impersonal academic style. Avoid unnecessary anglicisms when a clear Russian term exists. Every numbered formula, figure, table, and listing must be referenced from the text. Keep terminology, captions, section wording, and descriptions of repeated concepts stylistically consistent across the whole diploma.

## Documentation & Handoff Rules

`README.md` is the main project document and must always explain the project goal, local startup steps, and module overview.

`HADNOFF.md` must live in the repository root and be created or updated at the end of every task. Keep it brief: summarize recent work, include the latest user requests, and leave enough context for the next agent to continue from `README.md` and the handoff file.

- Local path: `/Users/margasanov/bmstu/8sem/diplom`
- Remote path: `/home/margasanov/temki/diplom`
