@echo off
title OVAscan AI - Public Web Deployment
echo ========================================================
echo   OVAscan AI - 1-Click Public Web Deployment
echo ========================================================
echo.
echo 1. Launching local AI server (port 5050)...
start /b python app.py
timeout /t 3 >nul
echo 2. Launching Cloudflare Secure Public Tunnel...
echo Look for your public HTTPS link below:
echo.
.\cloudflared.exe tunnel --url http://localhost:5050
pause
