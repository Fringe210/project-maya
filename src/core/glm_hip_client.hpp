#pragma once
#include "strata/core/glm_hip_helper.h"
#include <string>
#include <mutex>
#include <cstdlib>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
struct GlmHipHelper {
    glmh* h = nullptr;
    std::mutex upload_mu;
#if defined(_WIN32)
    HMODULE mod = nullptr;
    DLL_DIRECTORY_COOKIE hip_dir = nullptr;
#endif
    decltype(&glmh_open) open = nullptr;
    decltype(&glmh_reserve) reserve = nullptr;
    decltype(&glmh_upload) upload = nullptr;
    decltype(&glmh_compute) compute = nullptr;
    decltype(&glmh_close) close = nullptr;
    decltype(&glmh_read_q8) read_q8 = nullptr;
    decltype(&glmh_read_bytes) read_bytes = nullptr;
    bool load(const std::string& path, std::string& err) {
#if defined(_WIN32) && !defined(STRATA_USE_HIP)
        const char* bin=std::getenv("STRATA_GLM_HIP_HELPER_BIN");
        if (!bin) bin=std::getenv("MAYA_GLM_HIP_HELPER_BIN");
        std::string hipbin=bin ? bin : "C:/Program Files/AMD/ROCm/6.4/bin";
        std::wstring wide(hipbin.begin(),hipbin.end());
        hip_dir=AddDllDirectory(wide.c_str());
        mod=LoadLibraryExA(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!mod) { err="HIP helper DLL load failed: " + path + " (Windows " + std::to_string(GetLastError()) + ")"; return false; }
        auto version=(int(*)())GetProcAddress(mod,"glmh_version");
        if (!version || version()!=1) { err="GLM HIP helper ABI version mismatch"; return false; }
        open=(decltype(open))GetProcAddress(mod,"glmh_open");
        reserve=(decltype(reserve))GetProcAddress(mod,"glmh_reserve");
        upload=(decltype(upload))GetProcAddress(mod,"glmh_upload");
        compute=(decltype(compute))GetProcAddress(mod,"glmh_compute");
        close=(decltype(close))GetProcAddress(mod,"glmh_close");
        read_q8=(decltype(read_q8))GetProcAddress(mod,"glmh_read_q8");
        read_bytes=(decltype(read_bytes))GetProcAddress(mod,"glmh_read_bytes");
        if (!open || !reserve || !upload || !compute || !close) { err="GLM HIP helper ABI incomplete"; return false; }
        return true;
#else
        err="cross-vendor GLM helper requires Windows CUDA primary"; return false;
#endif
    }
    ~GlmHipHelper() {
        if (h && close) close(h);
#if defined(_WIN32)
        if (mod) FreeLibrary(mod);
        if (hip_dir) RemoveDllDirectory(hip_dir);
#endif
    }
};
