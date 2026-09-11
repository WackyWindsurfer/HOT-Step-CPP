#pragma once
// Is the TensorRT runtime actually loadable on this machine?
//
// On Windows the engine links nvinfer_10 / nvonnxparser_10 with /DELAYLOAD
// (engine/CMakeLists.txt), so the executable starts without the DLLs and the
// Model Manager downloads them next to it on demand. The price is that the
// first call into a missing DLL raises a delay-load exception, so every
// TensorRT entry point must be gated on this probe first. The builder resource
// DLL (nvinfer_builder_resource_sm<cc>_10.dll) is only needed to BUILD an
// engine; nvinfer looks it up itself, this just reports whether it is there.
//
// On Linux the libraries are ordinary link-time dependencies: if the binary
// runs, they are present.

#include <string>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

struct HotStepTrtRuntimeProbe {
    bool nvinfer          = false;
    bool parser           = false;
    bool builder_resource = false;
};

// sm = compute capability major*10+minor of the device engines are built for.
static inline HotStepTrtRuntimeProbe hot_step_trt_runtime_probe(int sm) {
    HotStepTrtRuntimeProbe p;
#ifdef HOT_STEP_TRT
#    ifdef _WIN32
    auto loadable = [](const wchar_t * name) {
        // Same search order the delay-load helper uses (application directory
        // first), so "probe says yes" and "call succeeds" agree.
        HMODULE h = LoadLibraryW(name);
        if (!h) {
            return false;
        }
        FreeLibrary(h);
        return true;
    };
    p.nvinfer = loadable(L"nvinfer_10.dll");
    p.parser  = loadable(L"nvonnxparser_10.dll");
    const std::wstring exact = L"nvinfer_builder_resource_sm" + std::to_wstring(sm) + L"_10.dll";
    p.builder_resource = loadable(exact.c_str()) || loadable(L"nvinfer_builder_resource_ptx_10.dll");
#    else
    (void) sm;
    p.nvinfer = p.parser = p.builder_resource = true;
#    endif
#else
    (void) sm;
#endif
    return p;
}
