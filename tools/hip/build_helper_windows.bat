@echo off
setlocal
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined DS4_HIP_SDK set "DS4_HIP_SDK=C:\Program Files\AMD\ROCm\6.4"
if not defined DS4_HIP_ARCH set "DS4_HIP_ARCH=gfx1100"
set "HIP_SDK_F=%DS4_HIP_SDK:\=/%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "HIP_PATH=%DS4_HIP_SDK%"
set "HIP_PLATFORM=amd"
set "PATH=%DS4_HIP_SDK%\bin;%SRC%\.venv\Scripts;%PATH%"
cmake -G Ninja -S "%SRC%" -B "%SRC%/build-hip" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_DS4_HELPER=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_NATIVE_EXPERTS=ON -DCMAKE_HIP_ARCHITECTURES=%DS4_HIP_ARCH% "-DCMAKE_C_COMPILER=%HIP_SDK_F%/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%HIP_SDK_F%/bin/clang++.exe" "-DCMAKE_HIP_COMPILER=%HIP_SDK_F%/bin/clang++.exe" "-DCMAKE_PREFIX_PATH=%HIP_SDK_F%" "-DSTRATA_GGML_DIR=%SRC%/third_party/llama.cpp"
if errorlevel 1 exit /b 1
cmake --build "%SRC%/build-hip" --target maya-hhelper -j 4
exit /b %errorlevel%
