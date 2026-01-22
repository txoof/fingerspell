# fingerspell
Finger Spelling Practice in Python





## Packaging Instructions

This guide covers building standalone executables for macOS and Windows.

### Prerequisites

Install PyInstaller in your virtual environment:

```bash
pip install pyinstaller
```

### Building the Application

#### macOS Build

From the project root directory:

```bash
pyinstaller fingerspell.spec
```

This creates:
- `dist/Fingerspell/` - Folder containing the executable and dependencies
- `dist/Fingerspell.app` - macOS application bundle

#### Windows Build

From the project root directory:

```bash
pyinstaller fingerspell.spec
```

This creates:
- `dist\Fingerspell\` - Folder containing the executable and dependencies

### Testing the Build

#### Test 1: Verify Models Are Bundled

Check that the models directory exists in the output:

**macOS:**
```bash
ls -la dist/Fingerspell/models/
```

**Windows:**
```cmd
dir dist\Fingerspell\models\
```

You should see:
- `ngt_static_classifier.pkl`
- `ngt_dynamic_classifier.pkl`

#### Test 2: Run the Executable

**macOS (folder mode):**
```bash
cd dist/Fingerspell
./Fingerspell
```

**macOS (app bundle):**
```bash
open dist/Fingerspell.app
```

**Windows:**
```cmd
cd dist\Fingerspell
Fingerspell.exe
```

#### Test 3: Check Resource Loading

The application should:
1. Start without errors about missing models
2. Display the camera feed
3. Recognize hand signs

If you see "ERROR: Model not found", the bundling failed.

### Troubleshooting

#### Issue: Models Not Found

Check the spec file `datas` section includes:
```python
datas=[('models/*.pkl', 'models')]
```

Rebuild with verbose output:
```bash
pyinstaller --log-level DEBUG fingerspell.spec
```

#### Issue: MediaPipe or OpenCV Errors

Add missing imports to `hiddenimports` in the spec file.

Common additions:
```python
hiddenimports=[
    'cv2',
    'mediapipe',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
]
```

#### Issue: macOS "App is damaged" Message

This happens because the app isn't signed. Users must:

1. Right-click the app
2. Select "Open"
3. Click "Open" in the security dialog

Or disable Gatekeeper temporarily:
```bash
xattr -cr dist/Fingerspell.app
```

#### Issue: Large File Size

The one-folder build includes all dependencies. Typical size: 200-400 MB.

To reduce size, exclude unnecessary packages in the spec file:
```python
excludes=['matplotlib', 'pandas', 'jupyter']
```

### Distribution

#### macOS

Zip the entire folder:
```bash
cd dist
zip -r Fingerspell-mac.zip Fingerspell/
```

Or create a DMG (requires additional tools).

#### Windows

Zip the entire folder:
```cmd
cd dist
powershell Compress-Archive Fingerspell Fingerspell-windows.zip
```

Or use an installer creator like Inno Setup.

### Quick Smoke Test Script

Create `test_packaged.py` to verify the build:

```python
#!/usr/bin/env python3
"""Quick test that models are accessible."""

from src.fingerspell.utils import verify_resource_exists

models = [
    'models/ngt_static_classifier.pkl',
    'models/ngt_dynamic_classifier.pkl'
]

print("Checking bundled resources...")
for model in models:
    exists = verify_resource_exists(model)
    status = "✓" if exists else "✗"
    print(f"{status} {model}")

if all(verify_resource_exists(m) for m in models):
    print("\nAll resources found. Build is good.")
    exit(0)
else:
    print("\nMissing resources. Build failed.")
    exit(1)
```

Run after building:
```bash
cd dist/Fingerspell
./Fingerspell --test  # Add this flag to your entry point for testing
```