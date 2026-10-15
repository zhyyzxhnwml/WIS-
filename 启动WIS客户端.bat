@echo off
chcp 65001 >nul
cd /d "%~dp0"
"E:\ACACONDA\envs\sam3\python.exe" -X utf8 -m train.wis.serve --open
if errorlevel 1 pause