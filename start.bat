@echo off
REM ============================================================
REM  RAG System — Windows Startup Script
REM ============================================================
setlocal enabledelayedexpansion

REM ── Resolve project root ──────────────────────────────────
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

REM ── Banner ────────────────────────────────────────────────
echo.
echo ======================================================
echo           RAG Chatbot System  v1.0
echo    PDF Upload . Chunking . Embedding . Chat
echo ======================================================
echo.

REM ── 1. Check Python version (>= 3.10) ────────────────────
set "PYTHON="

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PY_VER=%%v"
    for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
        if %%a GEQ 3 if %%b GEQ 10 set "PYTHON=python"
    )
)

if not defined PYTHON (
    where python3 >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        for /f "tokens=*" %%v in ('python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "PY_VER=%%v"
        for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
            if %%a GEQ 3 if %%b GEQ 10 set "PYTHON=python3"
        )
    )
)

if not defined PYTHON (
    echo [ERROR] Python 3.10 or higher is required but was not found.
    echo         Please install Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('%PYTHON% --version 2^>^&1') do set "PY_FULL_VER=%%v"
echo [OK] Found %PY_FULL_VER%

REM ── 2. Create virtual environment if it does not exist ────
set "VENV_DIR=%SCRIPT_DIR%venv"
if not exist "%VENV_DIR%\" (
    echo [..] Creating virtual environment ...
    %PYTHON% -m venv "%VENV_DIR%"
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM ── 3. Activate the virtual environment ───────────────────
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Virtual environment activated

REM ── 4. Install / update dependencies ─────────────────────
set "INSTALLED_FLAG=%VENV_DIR%\.installed"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"
set "NEEDS_INSTALL=0"

if not exist "%INSTALLED_FLAG%" (
    set "NEEDS_INSTALL=1"
) else (
    REM Check if requirements.txt is newer than .installed flag
    for %%R in ("%REQ_FILE%") do set "REQ_DATE=%%~tR"
    for %%I in ("%INSTALLED_FLAG%") do set "FLAG_DATE=%%~tI"
    REM Simple date comparison — reinstall if flag is older
    if "!REQ_DATE!" gtr "!FLAG_DATE!" (
        echo [..] requirements.txt changed since last install
        set "NEEDS_INSTALL=1"
    )
)

if "!NEEDS_INSTALL!"=="1" (
    echo [..] Installing Python dependencies (this may take a few minutes the first time) ...
    pip install --upgrade pip setuptools wheel -q
    pip install -r "%REQ_FILE%"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo. > "%INSTALLED_FLAG%"
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies up to date
)

REM ── 5. Create required directories ───────────────────────
for %%d in (uploads data models logs config) do (
    if not exist "%SCRIPT_DIR%%%d\" mkdir "%SCRIPT_DIR%%%d"
)
echo [OK] Required directories verified

REM ── 6. Copy .env.example to .env if .env is missing ──────
if not exist "%SCRIPT_DIR%.env" (
    if exist "%SCRIPT_DIR%.env.example" (
        copy "%SCRIPT_DIR%.env.example" "%SCRIPT_DIR%.env" >nul
        echo [!!] Created .env from .env.example — edit it to add your API keys
    )
)

REM ── 7. Read configuration defaults ───────────────────────
if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=8000"
if not defined LOG_LEVEL set "LOG_LEVEL=info"

REM Load .env file if it exists
if exist "%SCRIPT_DIR%.env" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%SCRIPT_DIR%.env") do (
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "%%b"=="" (
                set "%%a=%%b"
            )
        )
    )
)

if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=8000"
if not defined LOG_LEVEL set "LOG_LEVEL=info"

REM ── 8. Launch the server ──────────────────────────────────
echo.
echo ==========================================================
echo   RAG Chatbot Server — Starting ...
echo ----------------------------------------------------------
echo   Local URL :  http://127.0.0.1:%PORT%
echo   Network   :  http://%HOST%:%PORT%
echo   API Docs  :  http://127.0.0.1:%PORT%/docs
echo   Log Level :  %LOG_LEVEL%
echo ----------------------------------------------------------
echo   Press Ctrl+C to stop the server
echo ==========================================================
echo.

uvicorn app.main:app --host %HOST% --port %PORT% --log-level %LOG_LEVEL% --reload

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Server exited with an error. Check the logs above.
    pause
)

popd
endlocal
