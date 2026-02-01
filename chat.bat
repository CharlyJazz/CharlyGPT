@echo off
setlocal

REM Start backend (FastAPI)
start "sandbox_server" cmd /k python -m uvicorn sandbox_server.app:app --host 127.0.0.1 --port 8000 --reload

REM Start frontend (Vite)
start "sandbox_frontend" cmd /k npm run dev --prefix sandbox_frontend

endlocal
