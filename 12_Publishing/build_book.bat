:: build_book.bat
@echo off
where pandoc >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Pandoc is not installed.
    pause
    exit /b
)

pandoc title.txt ch01.md ch02.md ch03.md -o Grugs_Wisdom.epub --metadata title="Grug's Wisdom" --toc
echo Success if no errors above.
pause