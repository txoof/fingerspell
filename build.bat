@echo off
REM Build script for Fingerspell application
REM Usage: build.bat [--launch]

echo === Fingerspell Build Script ===
echo.

REM Check if pyinstaller is installed
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyInstaller not found
    echo Install with: pip install pyinstaller
    exit /b 1
)

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo [OK] Clean complete
echo.

REM Run tests
echo Running tests...
python -m pytest tests/test_resources.py -v
if errorlevel 1 (
    echo [FAIL] Tests failed
    exit /b 1
)
echo [OK] Tests passed
echo.

REM Build the application
echo Building application...
pyinstaller fingerspell.spec
if errorlevel 1 (
    echo [FAIL] Build failed
    exit /b 1
)
echo [OK] Build complete
echo.

REM Check build output
if exist "dist\Fingerspell" (
    echo Build successful!
    echo Location: dist\Fingerspell\
    echo.
    
    REM Launch if requested
    if "%1"=="--launch" (
        echo Launching application...
        cd dist\Fingerspell
        start Fingerspell.exe
    ) else (
        echo To launch: cd dist\Fingerspell ^&^& Fingerspell.exe
        echo Or run: build.bat --launch
    )
) else (
    echo ERROR: Build failed - no output found
    exit /b 1
)
