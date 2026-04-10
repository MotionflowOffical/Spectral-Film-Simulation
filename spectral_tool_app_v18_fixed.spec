# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

project_dir = Path.cwd()
script_name = "spectral_tool_app_v18_fixed.py"
script_path = project_dir / script_name

if not script_path.exists():
    raise SystemExit(
        f"Could not find {script_name} in {project_dir}. "
        "Place this .spec file in the same folder as the script before building."
    )

datas = []
binaries = []
hiddenimports = []

# Core UI / plotting support
datas += collect_data_files("matplotlib")
datas += collect_data_files("PIL")
hiddenimports += collect_submodules("matplotlib.backends")
hiddenimports += [
    "matplotlib",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.figure",
    "PIL.Image",
    "PIL.ImageOps",
    "PIL.ImageTk",
    "PIL._tkinter_finder",
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
]

# Optional packages used by the app. If installed, bundle them.
for pkg in ("rawpy", "tifffile"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

a = Analysis(
    [str(script_path)],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SpectralBandSplitter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SpectralBandSplitter",
)
