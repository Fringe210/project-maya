@echo off
setlocal
cd /d "%~dp0"
if not defined MAYA_CUDA_ARCHS set "MAYA_CUDA_ARCHS=120"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%CD%\.venv\Scripts;%PATH%"
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_NATIVE_EXPERTS=ON -DMAYA_BUILD_UNIFIED_BACKEND=ON -DCMAKE_CUDA_ARCHITECTURES=%MAYA_CUDA_ARCHS% "-DSTRATA_GGML_DIR=%CD%/third_party/llama.cpp"
if errorlevel 1 exit /b 1
cmake --build build --target maya-backend -j 8
exit /b %errorlevel%
