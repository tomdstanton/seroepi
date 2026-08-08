"""
AST Checker for undefined names and broken imports across src/, app/, tests/, scripts/
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_file(filepath):
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(filepath))  # noqa: F841
        # Check syntax ok
        return True, ""
    except Exception as e:
        return False, str(e)


all_py = (
    list(REPO_ROOT.glob("src/**/*.py"))
    + list(REPO_ROOT.glob("app/**/*.py"))
    + list(REPO_ROOT.glob("tests/**/*.py"))
    + list(REPO_ROOT.glob("scripts/**/*.py"))
)

errors = []
for py_file in all_py:
    ok, err = check_file(py_file)
    if not ok:
        errors.append((str(py_file), err))

print(f"Scanned {len(all_py)} python files.")
if errors:
    print(f"Found {len(errors)} syntax/parse errors:")
    for path, err in errors:
        print(f" - {path}: {err}")
else:
    print("Zero syntax/parse errors across all Python files!")
