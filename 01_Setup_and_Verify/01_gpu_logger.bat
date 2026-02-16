:: 01_gpu_logger.bat
:: Grug say: Watch the rock sweat.
@echo off
echo Logging GPU stats to ch01_gpu_log.csv...
echo Press Ctrl+C to stop.

echo timestamp, memory.used, memory.total > ch01_gpu_log.csv

:loop
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader >> ch01_gpu_log.csv
timeout /t 1 >nul
goto loop