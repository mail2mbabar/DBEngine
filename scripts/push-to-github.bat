@echo off
REM Adds Git + GitHub CLI to PATH for THIS window only, then pushes.
set "PATH=C:\Program Files\Git\cmd;C:\Program Files\GitHub CLI;%PATH%"

cd /d "%~dp0.."
echo Current folder:
cd

git --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: git not found. Install Git for Windows from https://git-scm.com/download/win
  pause
  exit /b 1
)

gh version >nul 2>&1
if errorlevel 1 (
  echo ERROR: gh not found. Expected at C:\Program Files\GitHub CLI\gh.exe
  pause
  exit /b 1
)

echo.
echo --- git status ---
git status -sb

echo.
echo --- Creating repo and pushing (mail2mbabar/DBEngine) ---
gh repo create DBEngine --public --source=. --remote=origin --push

if errorlevel 1 (
  echo.
  echo If the repo already exists, try:
  echo   git remote remove origin
  echo   git remote add origin https://github.com/mail2mbabar/DBEngine.git
  echo   git push -u origin main
)

pause
