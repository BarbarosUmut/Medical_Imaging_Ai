# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Medical Imaging AI.

Build:
    pyinstaller medical_imaging_ai.spec --clean --noconfirm

Output:
    dist/MedicalImagingAI/         (folder containing .exe + dependencies)
    dist/MedicalImagingAI/MedicalImagingAI.exe   (entry point)

Notes:
- --onedir mode (faster startup, smaller risk of antivirus false positives).
- CUDA torch DLLs are bundled (large output ~3-4 GB).
- Models are bundled into models/ next to the exe via Tree().
- Gradio/MONAI/nibabel dynamic-import quirks handled via collect_all().
- Ollama is NOT bundled — users must install it separately.
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# ---------------------------------------------------------------------------
# Collect dynamic-import-heavy packages
# ---------------------------------------------------------------------------

# Gradio: heavy frontend assets + dynamic component loading.
gradio_datas, gradio_binaries, gradio_hidden = collect_all("gradio")
gradio_client_datas, gradio_client_binaries, gradio_client_hidden = collect_all("gradio_client")

# MONAI: dynamic plugin loading.
monai_datas, monai_binaries, monai_hidden = collect_all("monai")

# nibabel: NIfTI parsers.
nibabel_datas, nibabel_binaries, nibabel_hidden = collect_all("nibabel")

# pydicom: codec plugins.
pydicom_datas, pydicom_binaries, pydicom_hidden = collect_all("pydicom")

# safehttpx: reads version.txt at import time — needs collect_all, not just hidden.
safehttpx_datas, safehttpx_binaries, safehttpx_hidden = collect_all("safehttpx")

# groovy: gradio dep that may also ship data files.
try:
    groovy_datas, groovy_binaries, groovy_hidden = collect_all("groovy")
except Exception:
    groovy_datas, groovy_binaries, groovy_hidden = [], [], []

# tomlkit: gradio config parser, has data.
try:
    tomlkit_datas, tomlkit_binaries, tomlkit_hidden = collect_all("tomlkit")
except Exception:
    tomlkit_datas, tomlkit_binaries, tomlkit_hidden = [], [], []

# huggingface_hub: gradio uses it for downloads.
try:
    hf_datas, hf_binaries, hf_hidden = collect_all("huggingface_hub")
except Exception:
    hf_datas, hf_binaries, hf_hidden = [], [], []

# Misc hidden imports that PyInstaller doesn't auto-detect.
extra_hidden = [
    "starlette",
    "uvicorn",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.logging",
    "matplotlib.backends.backend_agg",
    "skimage",
    "scipy.special.cython_special",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    ["app/main.py"],
    pathex=["."],
    binaries=(
        gradio_binaries
        + gradio_client_binaries
        + monai_binaries
        + nibabel_binaries
        + pydicom_binaries
        + safehttpx_binaries
        + groovy_binaries
        + tomlkit_binaries
        + hf_binaries
    ),
    datas=(
        gradio_datas
        + gradio_client_datas
        + monai_datas
        + nibabel_datas
        + pydicom_datas
        + safehttpx_datas
        + groovy_datas
        + tomlkit_datas
        + hf_datas
        # Bundle our own packages and model weights as data so the
        # _MEIPASS-relative paths in app/main.py resolve correctly.
        + [
            ("models", "models"),
            ("inference", "inference"),
            ("llm", "llm"),
            ("utils", "utils"),
            ("app", "app"),
        ]
    ),
    hiddenimports=(
        gradio_hidden
        + gradio_client_hidden
        + monai_hidden
        + nibabel_hidden
        + pydicom_hidden
        + safehttpx_hidden
        + groovy_hidden
        + tomlkit_hidden
        + hf_hidden
        + extra_hidden
        + collect_submodules("inference")
        + collect_submodules("llm")
        + collect_submodules("utils")
        + collect_submodules("app")
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim things we don't need to keep size sane.
    excludes=[
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "tensorboard",
        "tensorflow",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MedicalImagingAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX often corrupts torch DLLs — keep off.
    console=True,        # Keep a console window so users see startup logs.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="app/icon.ico",  # add later if you want a custom icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MedicalImagingAI",
)
