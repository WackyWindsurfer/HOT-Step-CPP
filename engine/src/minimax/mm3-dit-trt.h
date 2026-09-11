#pragma once
// Native MM3 DiT forward runtime. Sampling, CFG and stitching remain in MM3.
// The prepared engine is immutable; cache misses refit the selected GGUF weights.
#include "mm3-model.h"
#include "yyjson.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef HOT_STEP_TRT
#    include <cuda_runtime_api.h>
#    include <NvInfer.h>
#endif

static std::string mm3_trt_engine_path(const MM3Model & m) {
    const char * path = std::getenv("MM3_DIT_TRT_ENGINE");
    return path && *path ? path : (std::filesystem::path(m.models_dir) / "mm3" / "mm3-dit-trt.engine").string();
}

static bool mm3_trt_available(const MM3Model & m, std::string * reason) {
#ifdef HOT_STEP_TRT
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        if (reason) {
            *reason = "TensorRT requires an available NVIDIA CUDA device";
        }
        return false;
    }
    if (m.backend && std::string(ggml_backend_name(m.backend)).rfind("CUDA", 0) != 0) {
        if (reason) {
            *reason = "TensorRT requires the CUDA engine backend";
        }
        return false;
    }
    if (m.role_file[MM3_R_DIT].file_type != 1 && m.role_file[MM3_R_DIT].file_type != 0) {
        if (reason) {
            *reason = "Select the F16 DiT model to use TensorRT";
        }
        return false;
    }
    std::error_code ec;
    const auto      path = mm3_trt_engine_path(m);
    if (std::filesystem::is_regular_file(path, ec) && std::filesystem::is_regular_file(path + ".json", ec)) {
        return true;
    }
    if (reason) {
        *reason = "Prepare a native MM3 TensorRT engine and weight manifest first";
    }
#else
    (void) m;
    if (reason) {
        *reason = "This engine build does not include TensorRT";
    }
#endif
    return false;
}

#ifdef HOT_STEP_TRT
namespace mm3_trt {
static void check(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void cuda_check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Logger : nvinfer1::ILogger {
    void log(Severity severity, const char * message) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[MM3-TRT] %s\n", message);
        }
    }
};

struct Buffer {
    float * data = nullptr;

    void allocate(size_t count) { cuda_check(cudaMalloc((void **) &data, count * sizeof(float)), "allocate inputs"); }

    ~Buffer() {
        if (data) {
            cudaFree(data);
        }
    }
};

struct DeviceScope {
    int previous = 0;

    explicit DeviceScope(int device) {
        cuda_check(cudaGetDevice(&previous), "get device");
        cuda_check(cudaSetDevice(device), "set device");
    }

    ~DeviceScope() { cudaSetDevice(previous); }
};

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return uint16_t((bits >> 16) | 0x40U);
    }
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

class Runtime final : public MM3DitRuntime {
    Logger                                       logger;
    int                                          device = 0;
    cudaStream_t                                 stream = nullptr;
    std::unique_ptr<nvinfer1::IRuntime>          runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    Buffer                                       dx, dc, dt, drc, drs, dy;
    std::vector<float>                           basis, condition, bx, bc, fourier, cosine, sine, velocity;
    int                                          cached_L       = 0;
    int                                          uploaded_B     = 0;
    float                                        uploaded_gate  = -1.0f;
    size_t                                       resident_bytes = 0;

    std::filesystem::path cache_path(const MM3Model & m, const std::string & source) {
        // File identity matches the model/adapter cache convention elsewhere in
        // MM3. Sampler, duration, seed and conditioning deliberately stay out.
        std::ostringstream identity;
        auto               add_file = [&](const std::string & name) {
            const auto path = std::filesystem::absolute(name).lexically_normal();
            identity << path.string() << '\n'
                     << std::filesystem::file_size(path) << '\n'
                     << std::filesystem::last_write_time(path).time_since_epoch().count() << '\n';
        };
        identity << "mm3-trt-refit-cache-v1\n"
                 << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << '\n';
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device), "identify GPU");
        int driver = 0;
        cuda_check(cudaDriverGetVersion(&driver), "identify driver");
        identity << properties.name << ':' << properties.major << ':' << properties.minor << ':' << driver << '\n';
        add_file(source);
        add_file(source + ".json");
        add_file(m.role_file[MM3_R_DIT].path);
        identity << m.rest_adapter_desc << '\n';
        // Keep the exact requested scale too: rest_adapter_desc is formatted
        // for display and rounds floats to six decimal places.
        for (const char * key : { "MM3_ADAPTER", "MM3_ADAPTER_SCALE" }) {
            const char * value = std::getenv(key);
            identity << key << '=' << (value ? value : "") << '\n';
        }
        size_t begin = 0;
        while (begin < m.rest_adapter_desc.size()) {
            const auto end   = m.rest_adapter_desc.find("; ", begin);
            const auto entry = m.rest_adapter_desc.substr(begin, end == std::string::npos ? end : end - begin);
            const auto at    = entry.rfind('@');
            check(at != std::string::npos, "Invalid merged adapter identity");
            const std::filesystem::path adapter(entry.substr(0, at));
            const bool                  directory = std::filesystem::is_directory(adapter);
            const auto                  weights   = directory ? adapter / "adapter_model.safetensors" : adapter;
            const auto                  config = (directory ? adapter : adapter.parent_path()) / "adapter_config.json";
            add_file(weights.string());
            identity << "adapter_config=" << config.string() << '\n';
            if (std::filesystem::is_regular_file(config)) {
                add_file(config.string());
                // Alpha is a merge input. Hash the small config contents as well
                // as metadata, including when a tool preserves its timestamp.
                std::ifstream config_file(config, std::ios::binary);
                check(bool(config_file), "Cannot read adapter configuration for cache identity");
                identity << config_file.rdbuf() << '\n';
            } else {
                identity << "absent\n";
            }
            if (end == std::string::npos) {
                break;
            }
            begin = end + 2;
        }
        uint64_t hash = 14695981039346656037ULL;
        for (unsigned char byte : identity.str()) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        std::ostringstream name;
        name << std::hex << hash << ".engine";
        return std::filesystem::path(source).parent_path() / "mm3-trt-cache" / name.str();
    }

    bool deserialize(const std::filesystem::path & path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0) {
            return false;
        }
        const auto        length = file.tellg();
        std::vector<char> bytes(size_t(length), 0);
        file.seekg(0);
        file.read(bytes.data(), length);
        if (!file) {
            return false;
        }
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        return bool(engine);
    }

    void save_cache(const std::filesystem::path & path) {
        std::filesystem::path temporary;
        try {
            std::filesystem::create_directories(path.parent_path());
            std::unique_ptr<nvinfer1::IHostMemory> bytes(engine->serialize());
            check(bool(bytes), "Cannot serialize refitted engine");
            check(std::filesystem::space(path.parent_path()).available > bytes->size() + 256ULL * 1024 * 1024,
                  "Insufficient space for native engine cache");
            temporary =
                path.string() + ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output.write(static_cast<const char *>(bytes->data()), std::streamsize(bytes->size()));
            output.close();
            check(bool(output), "Cannot write native engine cache");
            // The deterministic destination is immutable. A racing preparation
            // can win without invalidating this process's already-loaded model.
            if (!std::filesystem::exists(path)) {
                std::filesystem::rename(temporary, path);
            } else {
                std::filesystem::remove(temporary);
            }
            fprintf(stderr, "[MM3-TRT] Saved refitted weight cache: %s\n", path.string().c_str());
        } catch (const std::exception & e) {
            std::error_code ec;
            if (!temporary.empty()) {
                std::filesystem::remove(temporary, ec);
            }
            fprintf(stderr, "[MM3-TRT] Weight cache unavailable: %s (rendering remains available)\n", e.what());
        }
    }

    void refit(const MM3Model & m, const std::string & path) {
        yyjson_read_err                                         json_error{};
        std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
            yyjson_read_file((path + ".json").c_str(), 0, nullptr, &json_error), yyjson_doc_free);
        check(bool(doc), "Cannot read TensorRT weight manifest");
        auto root = yyjson_doc_get_root(doc.get());
        check(yyjson_get_int(yyjson_obj_get(root, "version")) == 1, "Unsupported TensorRT manifest version");
        const char * contract = yyjson_get_str(yyjson_obj_get(root, "contract"));
        check(contract && std::string(contract) == "mm3-dit-bf16-fp32-rope-v1", "Wrong TensorRT model contract");
        auto weights = yyjson_obj_get(root, "weights");
        check(yyjson_is_arr(weights) && yyjson_arr_size(weights) == 368, "Expected all 368 DiT parameters in manifest");
        std::unique_ptr<nvinfer1::IRefitter> refitter(nvinfer1::createInferRefitter(*engine, logger));
        check(bool(refitter), "Engine is not refittable");
        std::set<std::string>              seen_gguf, seen_engine;
        std::vector<std::vector<uint16_t>> storage;
        storage.reserve(368);
        size_t       index, count;
        yyjson_val * entry;
        yyjson_arr_foreach(weights, index, count, entry) {
            const char * source          = yyjson_get_str(yyjson_obj_get(entry, "gguf_name"));
            const char * target          = yyjson_get_str(yyjson_obj_get(entry, "engine_name"));
            auto         transpose_value = yyjson_obj_get(entry, "transpose");
            check(source && target && yyjson_is_bool(transpose_value), "Invalid weight mapping");
            check(seen_gguf.insert(source).second && seen_engine.insert(target).second, "Duplicate weight mapping");
            auto tensor = ggml_get_tensor(m.wctx_dit_cpu.ctx, source);
            check(tensor != nullptr, std::string("Missing DiT tensor: ") + source);
            check(tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_F32,
                  "TensorRT requires F16/F32 DiT weights");
            const size_t n         = size_t(ggml_nelements(tensor));
            const auto   prototype = refitter->getWeightsPrototype(target);
            check(prototype.type == nvinfer1::DataType::kBF16 && prototype.count == int64_t(n),
                  std::string("Engine weight shape/type mismatch: ") + target);
            std::vector<float> values(n);
            if (tensor->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(tensor, values.data(), 0, n * 4);
            } else {
                std::vector<ggml_fp16_t> half(n);
                ggml_backend_tensor_get(tensor, half.data(), 0, n * 2);
                ggml_fp16_to_fp32_row(half.data(), values.data(), int64_t(n));
            }
            storage.emplace_back(n);
            auto & dest = storage.back();
            if (yyjson_get_bool(transpose_value)) {
                check(tensor->ne[2] == 1 && tensor->ne[3] == 1, "Only matrix weights can be transposed");
                const int64_t in = tensor->ne[0], out = tensor->ne[1];
#    pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < in; ++i) {
                    for (int64_t o = 0; o < out; ++o) {
                        dest[size_t(i * out + o)] = bf16(values[size_t(o * in + i)]);
                    }
                }
            } else {
#    pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < int64_t(n); ++i) {
                    dest[size_t(i)] = bf16(values[size_t(i)]);
                }
            }
            check(refitter->setNamedWeights(target, { nvinfer1::DataType::kBF16, dest.data(), int64_t(n) }),
                  std::string("Cannot refit ") + target);
        }
        for (auto tensor = ggml_get_first_tensor(m.wctx_dit_cpu.ctx); tensor;
             tensor      = ggml_get_next_tensor(m.wctx_dit_cpu.ctx, tensor)) {
            const std::string name = ggml_get_name(tensor);
            if (name != "dit.rope_inv_freq" && name != "dit.time_fourier.weight") {
                check(seen_gguf.count(name) == 1, "Unmapped DiT parameter: " + name);
            }
        }
        check(refitter->getMissingWeights(0, nullptr) == 0, "Incomplete refit weights");
        check(refitter->refitCudaEngine(), "DiT weight refit failed");
    }
  public:
    size_t gpu_bytes() const override { return resident_bytes; }

    ~Runtime() override {
        // CUDA current-device state belongs to each host thread.
        int old = 0;
        cudaGetDevice(&old);
        cudaSetDevice(device);
        if (stream) {
            cudaStreamSynchronize(stream);
        }
        context.reset();
        engine.reset();
        runtime.reset();
        for (Buffer * b : { &dx, &dc, &dt, &drc, &drs, &dy }) {
            if (b->data) {
                cudaFree(b->data);
            }
            b->data = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
        cudaSetDevice(old);
    }

    void load(const MM3Model & m) {
        const auto   start        = std::chrono::steady_clock::now();
        const char * backend_name = ggml_backend_name(m.backend);
        check(backend_name && sscanf(backend_name, "CUDA%d", &device) == 1, "TensorRT requires a CUDA backend");
        DeviceScope  device_scope(device);
        const auto & c = m.synth_cfg.dit;
        check(c.block_count == 36 && c.embedding_length == 2048 && c.head_count == 32 && c.head_dim == 64 &&
                  c.in_channels == 128 && c.condition_dim == 2048 && c.ff_inner == 8192 && c.rope_dim == 32 &&
                  c.rope_theta == 10000 && c.fourier_dim == 256 && c.rope_type == "neox" &&
                  c.glu_order == "value_gate" && c.concat_channels == 2304 && c.layer_norm_eps == 1e-5f &&
                  !c.attn_bias && c.timestep_token_prepended && c.pre_post_conv_residual,
              "Selected GGUF does not match the native MM3 DiT architecture");
        size_t free_before = 0, total = 0;
        cuda_check(cudaMemGetInfo(&free_before, &total), "query GPU memory");
        const auto    path = mm3_trt_engine_path(m);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        check(bool(file), "Cannot open prepared MM3 TensorRT engine: " + path);
        const auto length = file.tellg();
        check(length > 0, "Empty TensorRT engine");
        // Engine weights plus context/input headroom. Never request a WDDM spill.
        check(free_before > size_t(length) + 1024ULL * 1024 * 1024,
              "Insufficient dedicated GPU memory for MM3 TensorRT");
        runtime.reset(nvinfer1::createInferRuntime(logger));
        check(bool(runtime), "Cannot create TensorRT runtime");
        const auto prepared  = cache_path(m, path);
        bool       cache_hit = std::filesystem::is_regular_file(prepared) && deserialize(prepared);
        if (!cache_hit) {
            // A rejected file belongs to our derived cache, not the supplied
            // engine. Remove it so successful refitting can repair the cache.
            std::error_code ec;
            std::filesystem::remove(prepared, ec);
            deserialize(path);
        }
        check(bool(engine), "Cannot load TensorRT engine; prepare it for this GPU and TensorRT version");
        check(engine->getNbOptimizationProfiles() == 1 && engine->getNbIOTensors() == 6,
              "Wrong MM3 TensorRT engine interface");
        for (const char * name : { "x", "cond", "timestep_fourier", "rope_cos", "rope_sin", "output" }) {
            check(engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT, "Expected F32 engine interface");
        }
        const auto loaded = std::chrono::steady_clock::now();
        if (!cache_hit) {
            refit(m, path);
        }
        const auto refitted = std::chrono::steady_clock::now();
        if (!cache_hit) {
            save_cache(prepared);
        } else {
            fprintf(stderr, "[MM3-TRT] Reused refitted weight cache: %s\n", prepared.string().c_str());
        }
        basis.resize(128);
        check(m.synth.dit.time_fourier->type == GGML_TYPE_F32, "Fourier basis must remain F32");
        ggml_backend_tensor_get(m.synth.dit.time_fourier, basis.data(), 0, basis.size() * 4);
        cuda_check(cudaStreamCreate(&stream), "create stream");
        context.reset(engine->createExecutionContext());
        check(bool(context), "Cannot create TensorRT context");
        check(context->setOptimizationProfileAsync(0, stream), "Cannot select TensorRT profile");
        dx.allocate(2 * 128 * 689);
        dc.allocate(2 * 2048 * 689);
        dt.allocate(512);
        drc.allocate(690 * 32);
        drs.allocate(690 * 32);
        dy.allocate(2 * 128 * 689);
        check(context->setTensorAddress("x", dx.data) && context->setTensorAddress("cond", dc.data) &&
                  context->setTensorAddress("timestep_fourier", dt.data) &&
                  context->setTensorAddress("rope_cos", drc.data) && context->setTensorAddress("rope_sin", drs.data) &&
                  context->setTensorAddress("output", dy.data),
              "Cannot bind TensorRT inputs");
        cuda_check(cudaStreamSynchronize(stream), "finish setup");
        size_t free_after = 0;
        cuda_check(cudaMemGetInfo(&free_after, &total), "query resident memory");
        resident_bytes = free_before > free_after ? free_before - free_after : size_t(length);
        fprintf(
            stderr,
            "[MM3-TRT] Loaded cached engine in %.0f ms, refit selected DiT weights in %.0f ms, %.1f MiB GPU resident\n",
            std::chrono::duration<double, std::milli>(loaded - start).count(),
            std::chrono::duration<double, std::milli>(refitted - loaded).count(), double(resident_bytes) / 1048576);
    }

    bool run(const float * x,
             const float * cond,
             float         gate,
             float         t,
             int64_t       length,
             float *       out_c,
             float *       out_u,
             std::string * err) override {
        try {
            DeviceScope scope(device);
            check(length >= 3 && length <= 689 && std::isfinite(t) && std::isfinite(gate),
                  "Invalid MM3 TensorRT forward shape or timestep");
            const int    L = int(length), B = out_u ? 2 : 1;
            const size_t N = size_t(128) * L, C = size_t(2048) * L;
            const bool   shape_changed    = cached_L != L;
            const float  effective_gate   = B == 2 ? 1.0f : gate;
            const bool   upload_condition = cond || shape_changed || uploaded_B != B || uploaded_gate != effective_gate;
            if (cached_L != L) {
                condition.clear();
                cached_L = L;
            }
            if (cond) {
                condition.resize(C);
                for (int p = 0; p < L; ++p) {
                    for (int c = 0; c < 2048; ++c) {
                        condition[size_t(c) * L + p] = cond[size_t(p) * 2048 + c];
                    }
                }
            }
            check(condition.size() == C, "No condition supplied for this TensorRT window");
            bx.resize(B * N);
            if (upload_condition) {
                bc.assign(B * C, 0);
            }
            fourier.resize(B * 256);
            velocity.resize(B * N);
            for (int b = 0; b < B; ++b) {
                std::copy_n(x, N, bx.data() + b * N);
            }
            if (upload_condition) {
                for (size_t i = 0; i < C; ++i) {
                    bc[i] = condition[i] * effective_gate;
                }
            }
            for (int b = 0; b < B; ++b) {
                for (int j = 0; j < 128; ++j) {
                    const double angle         = 2.0 * 3.14159265358979323846 * double(t) * double(basis[j]);
                    fourier[b * 256 + j]       = float(std::cos(angle));
                    fourier[b * 256 + 128 + j] = float(std::sin(angle));
                }
            }
            if (shape_changed) {
                cosine.resize(size_t(L + 1) * 32);
                sine.resize(cosine.size());
                for (int p = 0; p <= L; ++p) {
                    for (int j = 0; j < 16; ++j) {
                        const double angle         = p / std::pow(10000.0, double(2 * j) / 32);
                        cosine[size_t(p) * 32 + j] = cosine[size_t(p) * 32 + j + 16] = float(std::cos(angle));
                        sine[size_t(p) * 32 + j] = sine[size_t(p) * 32 + j + 16] = float(std::sin(angle));
                    }
                }
            }
            check(context->setInputShape("x", nvinfer1::Dims3(B, 128, L)) &&
                      context->setInputShape("cond", nvinfer1::Dims3(B, 2048, L)) &&
                      context->setInputShape("timestep_fourier", nvinfer1::Dims2(B, 256)) &&
                      context->setInputShape("rope_cos", nvinfer1::Dims2(L + 1, 32)) &&
                      context->setInputShape("rope_sin", nvinfer1::Dims2(L + 1, 32)),
                  "TensorRT profile does not support this window");
            const auto output_shape = context->getTensorShape("output");
            check(output_shape.nbDims == 3 && output_shape.d[0] == B && output_shape.d[1] == 128 &&
                      output_shape.d[2] == L,
                  "TensorRT output shape does not match the allocated velocity buffer");
            auto upload = [&](Buffer & buffer, const std::vector<float> & values) {
                cuda_check(
                    cudaMemcpyAsync(buffer.data, values.data(), values.size() * 4, cudaMemcpyHostToDevice, stream),
                    "upload inputs");
            };
            upload(dx, bx);
            if (upload_condition) {
                upload(dc, bc);
            }
            upload(dt, fourier);
            if (shape_changed) {
                upload(drc, cosine);
                upload(drs, sine);
            }
            check(context->enqueueV3(stream), "TensorRT DiT forward failed");
            cuda_check(cudaMemcpyAsync(velocity.data(), dy.data, velocity.size() * 4, cudaMemcpyDeviceToHost, stream),
                       "read velocity");
            cuda_check(cudaStreamSynchronize(stream), "DiT forward sync");
            for (float v : velocity) {
                if (!std::isfinite(v)) {
                    throw std::runtime_error("Non-finite TensorRT velocity");
                }
            }
            uploaded_B    = B;
            uploaded_gate = effective_gate;
            std::copy_n(velocity.data(), N, out_c);
            if (out_u) {
                std::copy_n(velocity.data() + N, N, out_u);
            }
            return true;
        } catch (const std::exception & e) {
            if (err) {
                *err = e.what();
            }
            return false;
        }
    }
};
}  // namespace mm3_trt
#endif

static bool mm3_trt_prepare(const MM3Model & m, std::string * err) {
    if (m.dit_runtime) {
        return true;
    }
#ifdef HOT_STEP_TRT
    try {
        auto value = std::make_shared<mm3_trt::Runtime>();
        value->load(m);
        m.dit_runtime = std::move(value);
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
#else
    if (err) {
        *err = "TensorRT is not available in this engine build";
    }
    return false;
#endif
}

static bool mm3_trt_run(const MM3Model & m,
                        const float *    x,
                        const float *    cond,
                        float            gate,
                        float            t,
                        int64_t          L,
                        float *          out_c,
                        float *          out_u,
                        std::string *    err) {
    if (!mm3_trt_prepare(m, err)) {
        return false;
    }
    const bool ok = m.dit_runtime->run(x, cond, gate, t, L, out_c, out_u, err);
    if (!ok) {
        m.dit_runtime.reset();
    }
    return ok;
}
