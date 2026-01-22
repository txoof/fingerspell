# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Fingerspell NGT Recognition

This spec file bundles:
- The fingerspell application
- Model files (.pkl)
- Required Python dependencies

Usage:
    pyinstaller fingerspell.spec
"""

import sys
from pathlib import Path

block_cipher = None

# Define what data files to include
added_files = [
    ('models/*.pkl', 'models'),  # Include all .pkl files from models/ directory
]

# Add MediaPipe model files
import mediapipe
mediapipe_path = Path(mediapipe.__file__).parent
added_files.append((str(mediapipe_path / 'modules'), 'mediapipe/modules'))

a = Analysis(
    ['fingerspell.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'sklearn.utils._cython_blas',
        'sklearn.tree._utils',
        'sklearn.ensemble._forest',
        'sklearn.tree',
        'sklearn.ensemble',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Only exclude TensorFlow since we don't use it for inference
        'tensorflow',
        'tensorboard',
        'tensorflow_metal',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Fingerspell',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for windowed app (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Fingerspell',
)

# For macOS app bundle (optional)
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Fingerspell.app',
        icon=None,
        bundle_identifier='com.fingerspell.ngt',
    )
