@echo off
cd /d "%~dp0"
set DATABASE_URL=postgresql://postgres:1234@localhost:5432/postgres
set TICK_SEC=3
python v2x_realtime_server.py %*
