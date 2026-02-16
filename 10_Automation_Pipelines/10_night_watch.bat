:: 10_night_watch.bat
@echo off
set INPUT_FILE=prompts.txt
set MODEL_NAME=qwen2.5

if not exist %INPUT_FILE% (
    echo Error: %INPUT_FILE% not found!
    pause
    exit /b
)

for /F "tokens=*" %%A in (%INPUT_FILE%) do (
    echo [Processing]: %%A
    echo PROMPT: %%A >> results_log.txt
    ollama run %MODEL_NAME% "%%A" >> results_log.txt
)
echo NIGHT WATCH FINISHED.
pause