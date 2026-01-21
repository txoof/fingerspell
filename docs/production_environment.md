# Fingerspell Production Environment Overview

## Purpose

This document describes the intended production environment for the Fingerspell project. It explains how the application code, supporting assets, machine learning models, and runtime configuration fit together, with an emphasis on maintainability, packaging, and deployment.

The production environment is designed to support local development, reproducible builds, and packaging into a single executable using PyInstaller or a similar tool.

## High Level Architecture

The Fingerspell system is composed of three major layers.

1. Application entry layer  
2. Library and domain logic layer  
3. Resource and artifact layer  

Each layer has a clear responsibility and minimal coupling to the others.

## Directory Structure Overview

The following directory tree shows the intended high level production layout. Only the most relevant directories are shown.

```
fingerspell/
├── app.py
├── README.md
├── LICENSE
├── pyproject.toml
├── src/
│   └── fingerspell/
│       ├── __init__.py
│       ├── main.py
│       ├── pipeline.py
│       ├── vision.py
│       ├── ml.py
│       └── utils.py
├── assets/
│   ├── videos/
│   └── images/
├── models/
│   └── trained/
├── data/
│   └── runtime/
├── docs/
├── build/
└── dist/
```

This structure separates production code, runtime assets, and development artifacts while remaining simple enough to package cleanly.

## Application Entry Layer

This layer is responsible for starting the application and handling user interaction.

app.py  
The primary entry point for execution. It performs minimal setup such as argument parsing or configuration loading and then delegates execution to the library code.

fingerspell.__main__  
Provides a module based entry point. This allows the application to be executed using the Python module interface and gives packaging tools a stable entry target.

The entry layer should contain no domain logic.

## Library and Domain Logic Layer

This layer lives under src/fingerspell and contains all reusable application logic.

src/fingerspell  
This is the main Python package. It uses a src based layout to ensure predictable imports and clean packaging.

pipeline.py  
High level orchestration logic. This coordinates video processing, model inference, and result handling.

vision.py  
Image and video related functionality such as frame extraction, preprocessing, and feature preparation.

ml.py  
Model loading and inference logic. Models are treated as external artifacts rather than embedded code.

utils.py  
Small shared helpers such as path resolution, validation, or timing utilities.

All production logic lives here. This layer must not depend on proof of concept code or experiments.

## Resource and Artifact Layer

This layer contains non code files required at runtime or build time.

assets  
Static media such as example videos or images. These may be bundled into the final executable if required.

models  
Trained machine learning models. These are loaded dynamically by the application and may be included or external depending on deployment needs.

data  
Runtime generated or intermediate data. This directory is not intended to be packaged into the final executable.

## Development and Experimentation Boundary

Experimental and exploratory work is intentionally separated from production code.

1. Proof of concept work lives outside src/fingerspell  
2. Experimental scripts or notebooks are never imported by the production package  
3. Production code is promoted by copying files into src, not by merging experimental history  

This approach ensures a clean and auditable production codebase.

## Packaging and Distribution

The structure is designed to support packaging into a standalone executable.

Key considerations

1. A single clear entry point  
2. Explicit inclusion of required assets and models  
3. No reliance on external file system layout assumptions  

The src based layout ensures deterministic imports and compatibility with packaging tools.

## Summary

The Fingerspell production environment separates entry logic, core application code, and supporting resources. This keeps the codebase clean, makes packaging predictable, and allows experimentation without contaminating production history.