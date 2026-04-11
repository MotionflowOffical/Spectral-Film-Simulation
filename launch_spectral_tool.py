#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "PIL": "pillow",
    "matplotlib": "matplotlib",
    "tifffile": "tifffile",
}

OPTIONAL_IMPORTS = {
    "rawpy": "rawpy",
}

DEFAULT_SCRIPT_NAME = "spectral_tool_app_v18_fixed.py"


def is_module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def install_packages(packages: list[str]) -> None:
    if not packages:
        return
    print("Installing missing packages: " + ", ".join(packages))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def ensure_dependencies(include_optional: bool = True) -> None:
    missing = [pkg for mod, pkg in REQUIRED_IMPORTS.items() if not is_module_available(mod)]
    if include_optional:
        missing += [pkg for mod, pkg in OPTIONAL_IMPORTS.items() if not is_module_available(mod)]

    deduped = []
    seen = set()
    for pkg in missing:
        if pkg not in seen:
            seen.add(pkg)
            deduped.append(pkg)

    if deduped:
        install_packages(deduped)
    else:
        print("All required dependencies are already installed.")


def resolve_script_path(script_arg: str | None) -> Path:
    script_name = script_arg or DEFAULT_SCRIPT_NAME
    base = Path(__file__).resolve().parent
    target = (base / script_name).resolve()
    if target.exists():
        return target
    raise FileNotFoundError(
        f"Could not find target script: {target}. "
        f"Place this launcher next to {DEFAULT_SCRIPT_NAME}, or pass --script."
    )


def launch_script(script_path: Path) -> int:
    print(f"Launching {script_path.name}...")
    return subprocess.call([sys.executable, str(script_path)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Spectral Band Splitter and install missing dependencies.")
    parser.add_argument("--script", type=str, default=None, help="Target Python file to launch.")
    parser.add_argument("--no-rawpy", action="store_true", help="Skip installing rawpy if CR3 support is not needed.")
    args = parser.parse_args()

    script_path = resolve_script_path(args.script)
    ensure_dependencies(include_optional=not args.no_rawpy)
    return launch_script(script_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Dependency installation failed with exit code {exc.returncode}.")
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"Launcher failed: {exc}")
        raise SystemExit(1)
