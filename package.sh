#!/usr/bin/env bash
# Package Fingerspell for distribution
# Creates a zip file with the app and user instructions

set -e

echo "=== Fingerspell Distribution Packager ==="
echo ""

# Check if app exists
if [ ! -d "dist/Fingerspell.app" ]; then
    echo "ERROR: Fingerspell.app not found in dist/"
    echo "Run ./build.sh first"
    exit 1
fi

# Create distribution directory
DIST_DIR="dist/Fingerspell-Distribution"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

echo "Preparing distribution package..."

# Copy the app
cp -R "dist/Fingerspell.app" "$DIST_DIR/"

# Copy user README (rename to README for visibility)
cp "USER_README.md" "$DIST_DIR/README.txt"

# Get version/date for filename
DATE=$(date +%Y%m%d)
ZIP_NAME="Fingerspell-mac-$DATE.zip"

# Create zip
echo "Creating zip file..."
cd dist
zip -r "$ZIP_NAME" "Fingerspell-Distribution/"
cd ..

# Calculate size
ZIP_SIZE=$(du -sh "dist/$ZIP_NAME" | cut -f1)

echo ""
echo "✓ Package created successfully!"
echo "File: dist/$ZIP_NAME"
echo "Size: $ZIP_SIZE"
echo ""
echo "Contents:"
echo "  - Fingerspell.app"
echo "  - README.txt (launch instructions)"
echo ""
echo "Ready to distribute!"

# Cleanup temp directory
rm -rf "$DIST_DIR"
