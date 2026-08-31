# -*- mode: python ; coding: utf-8 -*-

import os


a = Analysis(
    ['parakeet.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Parakeet is currently for local/personal use and CUDA/cuDNN are already
# installed on the machine. Do not bundle NVIDIA's large runtime DLLs; let
# ONNX Runtime load them from the system CUDA/cuDNN directories that
# parakeet.py preloads at startup.
#
# Keep ONNX Runtime's own CUDA provider DLLs (for example
# onnxruntime_providers_cuda.dll and onnxruntime_providers_shared.dll).

CUDA_DLL_PREFIXES = (
    "cublas",
    "cufft",
    "cudart",
    "curand",
    "cusolver",
    "cusparse",
    "nvrtc",
    "nvjitlink",
    "cudnn",
)


def keep_binary(entry):
    dest = str(entry[0]).lower().replace("\\", "/")
    basename = os.path.basename(dest)

    remove = (
        basename == "onnxruntime_providers_tensorrt.dll"
        or basename.startswith(CUDA_DLL_PREFIXES)
    )

    if remove:
        print("EXCLUDING:", dest)

    return not remove

a.binaries = [
    entry for entry in a.binaries
    if keep_binary(entry)
]

a.datas = [
    entry for entry in a.datas
    if keep_binary(entry)
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='parakeet',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='parakeet',
)
