@echo off
set SCRIPT_DIR=%~dp0
set HUB_HTML="%SCRIPT_DIR%docs\dashboard\design_lab.html"
echo [ANTIMATTER] Opening Design Lab & 3D Proving Ground Hub...
start "" %HUB_HTML%
exit /b 0
