@echo off 
ECHO Starting MCP Server and Gradio UI... 
cd /d "D:\mcp server\mcp_server" 
call venv\Scripts\activate 
start cmd /k "uvicorn main:app --host 0.0.0.0 --port 8000" 
timeout /t 5 
start cmd /k "python gradio_ui_dynamic.py" 
ECHO All services started. Press any key to exit... 
pause 
