@echo off
REM HOT-Step engine build - pinned to VS 2022 (VS 2026 MSVC crashes nvcc)
REM CUDA arch 120a = RTX 5090 (Blackwell); portable cmake bundled on PATH here
set "PATH=D:\AI\tools\cmake-3.31.12-windows-x86_64\bin;%PATH%"
set "HOT_STEP_CMAKE_FLAGS=-DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_NATIVE=OFF"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo VCVARS_FAILED
    exit /b 1
)
call D:\AI\HOT-Step-CPP\engine\build.cmd
exit /b %errorlevel%
