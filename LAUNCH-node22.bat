@echo off
REM LAUNCH-node22.bat - HOT-Step launcher pinned to the local Node 22 in .node22\
REM (system Node 24 breaks better-sqlite3; the project requires Node 18-22)
set "PATH=D:\AI\.node22\node-v22.18.0-win-x64;%PATH%"
call D:\AI\HOT-Step-CPP\LAUNCH.bat
