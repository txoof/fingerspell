#!/usr/bin/env bash
# Build script for Fingerspell application
# Usage: ./build.sh [--launch]

set -e  # Exit on error

echo "=== Fingerspell Build Script ==="
echo ""

# Check if pyinstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "ERROR: PyInstaller not found"
    echo "Install with: pip install pyinstaller"
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
echo "✓ Clean complete"
echo ""

# Run tests
echo "Running tests..."
if python -m pytest tests/test_resources.py -v; then
    echo "✓ Tests passed"
else
    echo "✗ Tests failed"
    exit 1
fi
echo ""

# Build the application
echo "Building application..."
pyinstaller fingerspell.spec
echo "✓ Build complete"
echo ""

# Check build output
if [ -d "dist/Fingerspell.app" ]; then
    BUILD_SIZE=$(du -sh dist/Fingerspell.app | cut -f1)
    echo "Build successful!"
    echo "Size: $BUILD_SIZE"
    echo "Location: dist/Fingerspell.app"
    echo ""
    
    # Launch if requested
    if [ "$1" == "--launch" ]; then
        echo "Launching application..."
        open dist/Fingerspell.app
    else
        echo "To launch: open dist/Fingerspell.app"
        echo "Or run: ./build.sh --launch"
    fi
elif [ -d "dist/Fingerspell" ]; then
    BUILD_SIZE=$(du -sh dist/Fingerspell | cut -f1)
    echo "Build successful!"
    echo "Size: $BUILD_SIZE"
    echo "Location: dist/Fingerspell/"
    echo ""
    
    # Launch if requested
    if [ "$1" == "--launch" ]; then
        echo "Launching application..."
        cd dist/Fingerspell
        ./Fingerspell
    else
        echo "To launch: cd dist/Fingerspell && ./Fingerspell"
        echo "Or run: ./build.sh --launch"
    fi
else
    echo "ERROR: Build failed - no output found"
    exit 1
fi
