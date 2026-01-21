# Fingerspell Distribution Guide For Non Technical End Users

## Goal

Deliver a double clickable application for macOS and Windows that requires zero Python knowledge. The client receives a zip file, unzips it, and runs the app.

Constraints

1. No Docker
2. No pip install
3. No remote servers
4. Must run on macOS and Windows
5. Total runtime assets around 30 to 60 MB excluding the Python environment

Recommended solution

1. Build a standalone app per platform using PyInstaller
2. Distribute as a folder based build zipped for delivery
3. Bundle required runtime assets and trained models into the build
4. Use a single path resolver in code so assets load correctly both in development and in the bundled executable

## High Level Packaging Approach

Why folder based distribution is preferred

1. ML dependencies and model artifacts load more reliably as separate files on disk
2. Faster startup than one file builds in many cases
3. Fewer issues with dynamic libraries and hidden imports
4. Easier to debug because files exist on disk

What the client receives

macOS

1. Fingerspell_Mac folder
2. Inside it, Fingerspell.app
3. Optional, samples and readme
4. Client action: unzip then double click Fingerspell.app

Windows

1. Fingerspell_Windows folder
2. Inside it, Fingerspell.exe
3. Optional, samples and readme
4. Client action: unzip then double click Fingerspell.exe

## Recommended Repo Layout For Shipping

Only relevant high level directories shown

```
fingerspell/
├── app.py
├── pyproject.toml
├── README.md
├── src/
│   └── fingerspell/
│       ├── init.py
│       ├── main.py
│       ├── paths.py
│       ├── pipeline.py
│       ├── ml.py
│       └── vision.py
├── assets/
│   ├── videos/
│   └── images/
├── models/
│   └── trained/
├── docs/
├── build/
└── dist/
```

Directory responsibilities

1. src/fingerspell  
   Production Python package. Everything importable and testable lives here.

2. assets  
   Runtime media that ships with the app, for example sample videos or reference images.

3. models  
   Trained model artifacts that the app loads at runtime.

4. build and dist  
   Build outputs. Not shipped in git. dist is what you zip for delivery.

## Critical Topic: Paths That Work In Development And In A Bundled App

The most common source of failure in packaged Python apps is file paths.

Wrong pattern

1. Loading resources relative to the current working directory
2. Assuming the app is always started from the repo root
3. Hard coding absolute paths from a developer machine

Instead, all code that loads assets or models should go through a single resolver function.

Design goals for paths

1. Works in normal development execution
2. Works in PyInstaller bundled execution
3. Supports both one folder builds and app bundles
4. Avoids relying on the current working directory
5. Allows writing logs and outputs to a user writable directory

## paths.py Reference Implementation

Place this file at `src/fingerspell/paths.py`

```python
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    """
    Resolved paths for both development and bundled execution.

    base_dir
      Root directory where bundled resources live at runtime.
      In development, this is the repository root.
      In PyInstaller, this is sys._MEIPASS.

    assets_dir
      Directory containing shipped assets such as sample videos or images.

    models_dir
      Directory containing shipped trained model artifacts.

    user_data_dir
      Writable directory for logs, outputs, caches, and runtime generated files.
    """
    base_dir: Path
    assets_dir: Path
    models_dir: Path
    user_data_dir: Path


def _is_bundled() -> bool:
    """
    Return True when running from a PyInstaller bundle.
    """
    return bool(getattr(sys, 'frozen', False)) and hasattr(sys, '_MEIPASS')


def _repo_root_from_here() -> Path:
    """
    Infer repository root from this file location.

    Assumes this file is at:
      repo_root / src / fingerspell / paths.py

    If you move things around, update this.
    """
    return Path(__file__).resolve().parents[2]


def _platform_user_data_dir(app_name: str = 'fingerspell') -> Path:
    """
    Return a user writable location for app outputs and logs.

    macOS
      ~/Library/Application Support/<app_name>

    Windows
      %APPDATA%\<app_name>

    Linux
      ~/.local/share/<app_name>
    """
    home = Path.home()

    if sys.platform == 'darwin':
        return home / 'Library' / 'Application Support' / app_name

    if sys.platform.startswith('win'):
        appdata = os.environ.get('APPDATA')
        if appdata:
            return Path(appdata) / app_name
        return home / app_name

    return home / '.local' / 'share' / app_name


def get_app_paths(app_name: str = 'fingerspell') -> AppPaths:
    """
    Compute and return all key application paths.

    Bundled mode
      base_dir is sys._MEIPASS

    Development mode
      base_dir is repository root inferred from file location
    """
    if _is_bundled():
        base_dir = Path(getattr(sys, '_MEIPASS')).resolve()
    else:
        base_dir = _repo_root_from_here()

    assets_dir = base_dir / 'assets'
    models_dir = base_dir / 'models'
    user_data_dir = _platform_user_data_dir(app_name=app_name)

    return AppPaths(
        base_dir=base_dir,
        assets_dir=assets_dir,
        models_dir=models_dir,
        user_data_dir=user_data_dir,
    )


def ensure_user_dirs(app_name: str = 'fingerspell') -> AppPaths:
    """
    Create the user writable directories if needed and return paths.

    This should be called early in app startup.
    """
    paths = get_app_paths(app_name=app_name)
    paths.user_data_dir.mkdir(parents=True, exist_ok=True)
    return paths


def asset_path(*parts: str, app_name: str = 'fingerspell') -> Path:
    """
    Convenience function to locate a shipped asset.
    """
    p = get_app_paths(app_name=app_name).assets_dir.joinpath(*parts)
    return p


def model_path(*parts: str, app_name: str = 'fingerspell') -> Path:
    """
    Convenience function to locate a shipped model artifact.
    """
    p = get_app_paths(app_name=app_name).models_dir.joinpath(*parts)
    return p


def user_path(*parts: str, app_name: str = 'fingerspell') -> Path:
    """
    Convenience function to locate a user writable output path.
    """
    p = ensure_user_dirs(app_name=app_name).user_data_dir.joinpath(*parts)
    return p
```

### How To Use paths.py

Import pattern

Use absolute imports inside the package:

```Python
from fingerspell.paths import asset_path, model_path, user_path, ensure_user_dirs
```

Call ensure_user_dirs near startup

Example in app.py:

```Python
from fingerspell.paths import ensure_user_dirs

def main() -> int:
    ensure_user_dirs()
    # parse args, call pipeline
    return 0
```

### Example 1: Load A Model

Assume model is located in `models/trained/model.tflite`.

Use:

```Python
from fingerspell.paths import model_path

model_file = model_path('trained', 'model.tflite')

# Example usage in a loader that expects a string path
model = load_model(str(model_file))
```

### Example 2: Load a Sample Video Asset

Assume the sample is stored in `assets/videos/sample_01.mp4`.

```Python
from fingerspell.paths import asset_path

sample_video = asset_path('videos', 'sample_01.mp4')

frames = extract_frames(str(sample_video))
```

### Example 3: Write Output and Logs

Never write logs or outputs into the application folder. Use a user-writable directory

Create a logs folder:

```Python
from fingerspell.paths import asset_path

sample_video = asset_path('videos', 'sample_01.mp4')

frames = extract_frames(str(sample_video))
```

Write an output file:

```Python
from fingerspell.paths import user_path

out_dir = user_path('outputs')
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / 'results.csv'
save_results_csv(out_csv)
```

In development, base_dir should be the repo root.

In a PyInstaller app, base_dir should point to the extracted bundle directory.

## PyInstaller Data Inclusion Notes

Your bundled app must explicitly include assets and models.

General approach

1. Create a PyInstaller spec file
2. Include assets and models using datas entries
3. Build on each target platform

Important detail

In PyInstaller one folder mode, your assets and models should be included so that they appear under the same base_dir that paths.py expects.

The implementation above expects these directories at runtime:

1. <base_dir>/assets
2. <base_dir>/models

If you include them into different locations, update paths.py accordingly.

### Platform Specific User Experience Notes

macOS

Unsigned apps often trigger Gatekeeper warnings.

Typical safe user instruction

1. Right click the app
2. Select Open
3. Confirm Open in the dialog

Windows

Unsigned executables may trigger SmartScreen.

Typical safe user instruction

1. Click More info
2. Click Run anyway

These steps are normal for university client pilots without code signing.

### Delivery Checklist

Before shipping

1. Build on macOS machine and Windows machine
2. Run a known sample video through the pipeline
3. Confirm output is created in the user data directory
4. Confirm log file is created
5. Zip the distribution folder

Zip content recommendation

macOS zip should contain

1. Fingerspell.app
2. README_RUN.txt
3. Optional samples folder if you want the client to have test videos

Windows zip should contain

1. Fingerspell.exe
2. README_RUN.txt
3. Optional samples folder

## Summary

To distribute Fingerspell to non technical users without Docker or Python, build a standalone PyInstaller app per platform and distribute as a zip. The most important implementation detail is path handling. Centralize all resource path resolution in src/fingerspell/paths.py and ensure assets and models are explicitly bundled so the runtime layout matches what the code expects.