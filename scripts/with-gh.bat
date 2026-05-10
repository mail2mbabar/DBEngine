@echo off
REM Prepends GitHub CLI to PATH, then runs gh with all arguments.
set "PATH=C:\Program Files\GitHub CLI;%PATH%"
cd /d "%~dp0.."
gh %*
