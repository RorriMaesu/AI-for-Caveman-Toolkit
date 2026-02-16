:: install_moss.bat
:: Grug say: Double click this to plant the Moss.
@echo off
echo --- GRUG'S ONE-CLICK MOSS INSTALLER ---

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: You need Git installed.
    pause
    exit /b
)

if exist "MOSS-TTS" (
    echo Folder already exists.
) else (
    git clone https://github.com/OpenMOSS/MOSS-TTS.git
)

cd MOSS-TTS
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo DONE.
pause