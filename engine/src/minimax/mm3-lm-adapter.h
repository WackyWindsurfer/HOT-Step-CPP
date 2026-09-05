#pragma once
// minimax/mm3-lm-adapter.h — runtime LoRA adapters for the MM3 global LM.
//
// HOT-Step file (does not exist upstream). Loads a PEFT-format LM LoRA
// (SimpleTuner `--minimax_music_train_component=language_model` checkpoints:
// keys `language_model.model.layers.N.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,
// down}_proj}.lora_{A,B}.weight`) and applies it as RUNTIME low-rank deltas
// inside mm3-lm-graph.h — the base weights are never modified, so per-group
// scales are live per generation with no reload.
//
// Why runtime deltas and not a load-time merge: the whole point (validated by
// the 2026-08-20 ablation grid on the alk3 r256 adapter) is DIALING groups at
// render time — attention carries the plan/genre, the MLPs carry vocal
// identity AND the fidelity damage, and attention 1.0 / MLP 0.5 was the ear
// winner. A merge would freeze one setting into 17 GB of resident weights.
//
// Scale model (multiplicative, all default 1.0):
//     effective(module, layer) = global
//                              * (attn | mlp)          by module kind
//                              * (early | mid | late)  by layer/12
// PEFT's own alpha/rank scaling: SimpleTuner LM checkpoints carry NO alpha
// tensors and train with alpha == rank, so the baked base scale is 1.0. If a
// per-module `.alpha` scalar IS present (comfy-style exports), it is honoured
// as alpha/rank.
//
// Cost: r256 f16 adapter ≈ 1.5 GB resident, streamed per AR token on top of
// the 17.2 GB base — expect ≈ +9 % on the LM step at rank 256, proportionally
// less at lower ranks. VRAM and step cost both scale linearly with rank.
//
// Scales are baked into cached graphs as constants; mm3-lm-graph.h owns an
// `adapter_epoch` and rebuilds its slots when (adapter, scales) change. That
// is once per generation, not per frame — rebuild cost is graph construction
// only.

#include "backend.h"
#include "safetensors.h"
#include "yyjson.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#define MM3_LM_ADAPTER_LAYERS  36
#define MM3_LM_ADAPTER_MODULES 7

enum MM3LmAdapterModule {
    MM3_LM_ADAPTER_Q = 0,
    MM3_LM_ADAPTER_K,
    MM3_LM_ADAPTER_V,
    MM3_LM_ADAPTER_O,
    MM3_LM_ADAPTER_GATE,
    MM3_LM_ADAPTER_UP,
    MM3_LM_ADAPTER_DOWN,
};

static const char * MM3_LM_ADAPTER_MODULE_KEY[MM3_LM_ADAPTER_MODULES] = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj",
};

struct MM3LmAdapterScales {
    float global = 1.0f;
    float attn   = 1.0f;
    float mlp    = 1.0f;
    float early  = 1.0f;  // layers 0..11
    float mid    = 1.0f;  // layers 12..23
    float late   = 1.0f;  // layers 24..35

    bool operator==(const MM3LmAdapterScales & o) const {
        return global == o.global && attn == o.attn && mlp == o.mlp && early == o.early && mid == o.mid &&
               late == o.late;
    }

    bool operator!=(const MM3LmAdapterScales & o) const { return !(*this == o); }
};

// Which component of a module a safetensors key carries. PEFT LoRA files use
// A/B; LyCORIS LoKr files use w1 plus either a monolithic w2 or the w2_a/w2_b
// pair, and a scalar alpha.
enum MM3LmAdapterComp {
    MM3_LM_COMP_NONE = 0,
    MM3_LM_COMP_A,
    MM3_LM_COMP_B,
    MM3_LM_COMP_W1,
    MM3_LM_COMP_W2,
    MM3_LM_COMP_W2A,
    MM3_LM_COMP_W2B,
    MM3_LM_COMP_ALPHA,
    // PEFT DoRA's lora_magnitude_vector, [out].
    MM3_LM_COMP_MAG,
    // LyCORIS LoHa. hada_w1_a is our B, hada_w1_b our A, hada_w2_a our B2 and
    // hada_w2_b our A2 (train/lm-export.h's LoHa branch, inverted).
    MM3_LM_COMP_HADA_W1A,
    MM3_LM_COMP_HADA_W1B,
    MM3_LM_COMP_HADA_W2A,
    MM3_LM_COMP_HADA_W2B,
};

struct MM3LmAdapterPair {
    ggml_tensor * a          = nullptr;  // [in, r]  (f16)
    ggml_tensor * b          = nullptr;  // [r, out] (f16)
    float         base_scale = 1.0f;     // alpha/rank when an alpha scalar exists, else 1

    // ── LoKr (dW = kron(w1, w2) * lokr_scale) ──────────────────────────────
    // ggml extents, mirroring lm_lokr_init: w1 [in_m, out_l], w2 [in_n, out_k],
    // w2_a [dim, out_k], w2_b [in_n, dim]. Geometry is read back off the
    // tensors so nothing here has to parse the LyCORIS config blob.
    ggml_tensor * w1         = nullptr;
    ggml_tensor * w2         = nullptr;  // monolithic
    ggml_tensor * w2_a       = nullptr;  // factorized
    ggml_tensor * w2_b       = nullptr;
    int64_t       in_m = 0, in_n = 0, out_l = 0, out_k = 0;
    float         lokr_scale = 1.0f;
    float         alpha_raw  = 0.0f;     // the file's alpha scalar, pre-division
    bool          has_alpha  = false;

    // ── DoRA (2026-09-05) ──────────────────────────────────────────────────
    // m   = PEFT's lora_magnitude_vector, straight off disk, [out] f32.
    // nrm = ||W + base_scale*BA||_col, which needs the BASE weights this loader
    //       never sees — mm3_lm_dora_prepare() fills it once the LM is resident.
    // Both present => mm3_lm_mm rescales by m/nrm. m without nrm is refused
    // before any graph is built (see MM3LmAdapter::dora_pending).
    ggml_tensor * m          = nullptr;
    ggml_tensor * nrm        = nullptr;

    // ── LoHa: delta = (A1 B1) (.) (A2 B2) ──────────────────────────────────
    // The second pair shares a/b's shapes. MERGE MODE ONLY — the delta is a
    // full [in, out] tensor, so a per-token runtime apply is not on.
    ggml_tensor * a2         = nullptr;  // [in, r]
    ggml_tensor * b2         = nullptr;  // [r, out]

    bool has_lora() const { return a && b; }
    bool has_lokr() const { return w1 && (w2 || (w2_a && w2_b)); }
    bool has_dora() const { return m && nrm; }
    bool has_loha() const { return a && b && a2 && b2; }
};

struct MM3LmAdapter {
    std::string      path;
    int64_t          mtime = 0;
    int              rank  = 0;   // LoRA only; 0 for LoKr
    bool             is_lokr = false;
    MM3LmAdapterPair mods[MM3_LM_ADAPTER_LAYERS][MM3_LM_ADAPTER_MODULES];
    int              n_loaded = 0;

    // ── adapter_config.json (2026-09-05) ───────────────────────────────────
    //
    // This loader used to read NOTHING but the tensors, taking base_scale from
    // an optional per-module `.alpha` scalar and defaulting to 1.0. That was
    // right for SimpleTuner checkpoints (no alpha tensors, alpha == rank) and
    // WRONG the moment a trainer wrote use_rslora: alpha/sqrt(r) is 16x
    // alpha/r at r256, and nothing in the file would have said so.
    //
    // COMPATIBILITY RULE, load-bearing: when adapter_config.json is absent, or
    // when it says alpha == r with use_rslora false, the computed base_scale is
    // exactly 1.0 — byte-identical to what every already-shipped adapter in
    // M:\HOT-Step-CPP\Adapters\mm3-lm-adapters got before this existed.
    bool             cfg_rslora = false;
    bool             cfg_dora   = false;
    std::string      cfg_peft   = "LORA";
    float            cfg_ratio  = -1.0f;  // alpha/r (or alpha/sqrt(r)); -1 = no config

    // Merge-only parameterizations. Both are LOADED here — merge mode needs the
    // tensors — and refused at the point of use by the runtime graph, which is
    // where the low-rank assumption actually lives.
    bool             is_loha      = false;
    bool             is_hira      = false;

    int              dora_n       = 0;
    bool             dora_pending = false;  // nrm not filled yet — do NOT build a graph
    // Which resident LM the norms were computed against, so a pristine reload
    // or a model swap re-runs the pass instead of reusing stale norms.
    const void *     dora_base    = nullptr;

    ggml_context *        ctx         = nullptr;
    ggml_backend_buffer_t buf         = nullptr;
    BackendPair           bp          = {};
    bool                  backend_ref = false;

    // effective scale for one module instance under the request's dials
    float effective(int layer, int module, const MM3LmAdapterScales & s) const {
        const MM3LmAdapterPair & p = mods[layer][module];
        float                    v = s.global * p.base_scale;
        v *= (module <= MM3_LM_ADAPTER_O) ? s.attn : s.mlp;
        v *= (layer < 12) ? s.early : (layer < 24) ? s.mid : s.late;
        return v;
    }
};

static void mm3_lm_adapter_free(MM3LmAdapter * ad) {
    if (!ad) {
        return;
    }
    if (ad->buf) {
        ggml_backend_buffer_free(ad->buf);
    }
    if (ad->ctx) {
        ggml_free(ad->ctx);
    }
    if (ad->backend_ref) {
        backend_release(ad->bp.backend, ad->bp.cpu_backend);
    }
    delete ad;
}

// Accept keys with or without the `language_model.` prefix, and PEFT's
// `.default.` infix (present after PeftModel round-trips).
static bool mm3_lm_adapter_parse_key(const std::string & key, int * layer, int * module,
                                     MM3LmAdapterComp * comp) {
    const char * s = key.c_str();
    if (strncmp(s, "language_model.", 15) == 0) {
        s += 15;
    }
    if (strncmp(s, "base_model.model.", 17) == 0) {
        s += 17;
    }
    if (strncmp(s, "model.layers.", 13) != 0) {
        return false;
    }
    s += 13;
    char * end   = nullptr;
    long   lyr   = strtol(s, &end, 10);
    if (end == s || *end != '.' || lyr < 0 || lyr >= MM3_LM_ADAPTER_LAYERS) {
        return false;
    }
    s = end + 1;
    int mod = -1;
    for (int m = 0; m < MM3_LM_ADAPTER_MODULES; m++) {
        size_t n = strlen(MM3_LM_ADAPTER_MODULE_KEY[m]);
        if (strncmp(s, MM3_LM_ADAPTER_MODULE_KEY[m], n) == 0 && s[n] == '.') {
            mod = m;
            s += n + 1;
            break;
        }
    }
    if (mod < 0) {
        return false;
    }
    // remainder: lora_A.weight / lora_B.weight, optionally lora_A.default.weight
    if (strncmp(s, "lora_A.", 7) == 0) {
        *comp = MM3_LM_COMP_A;
    } else if (strncmp(s, "lora_B.", 7) == 0) {
        *comp = MM3_LM_COMP_B;
    } else if (strncmp(s, "lora_magnitude_vector.", 22) == 0) {
        *comp = MM3_LM_COMP_MAG;
    } else {
        return false;
    }
    *layer  = (int) lyr;
    *module = mod;
    return true;
}

// LyCORIS LoKr keys, as written by lm_export_lokr:
//   lycoris_layers_<L>_<site with dots as underscores>.lokr_w1 | .lokr_w2
//                                                     | .lokr_w2_a | .lokr_w2_b
//                                                     | .alpha
static MM3LmAdapterComp mm3_lm_adapter_parse_lokr(const std::string & key, int * layer, int * module) {
    const char * s = key.c_str();
    if (strncmp(s, "lycoris_layers_", 15) != 0) {
        return MM3_LM_COMP_NONE;
    }
    s += 15;
    char * end = nullptr;
    long   lyr = strtol(s, &end, 10);
    if (end == s || *end != '_' || lyr < 0 || lyr >= MM3_LM_ADAPTER_LAYERS) {
        return MM3_LM_COMP_NONE;
    }
    s = end + 1;
    // Site name with '.' replaced by '_', so compare against the module keys
    // under the same substitution rather than keeping a second table.
    int mod = -1;
    for (int m = 0; m < MM3_LM_ADAPTER_MODULES; m++) {
        std::string want(MM3_LM_ADAPTER_MODULE_KEY[m]);
        for (size_t i = 0; i < want.size(); i++) {
            if (want[i] == '.') {
                want[i] = '_';
            }
        }
        if (strncmp(s, want.c_str(), want.size()) == 0 && s[want.size()] == '.') {
            mod = m;
            s += want.size() + 1;
            break;
        }
    }
    if (mod < 0) {
        return MM3_LM_COMP_NONE;
    }
    MM3LmAdapterComp c = MM3_LM_COMP_NONE;
    // Longest first: lokr_w2_a / lokr_w2_b must not be swallowed by lokr_w2.
    if (strcmp(s, "lokr_w2_a") == 0) {
        c = MM3_LM_COMP_W2A;
    } else if (strcmp(s, "lokr_w2_b") == 0) {
        c = MM3_LM_COMP_W2B;
    } else if (strcmp(s, "lokr_w1") == 0) {
        c = MM3_LM_COMP_W1;
    } else if (strcmp(s, "lokr_w2") == 0) {
        c = MM3_LM_COMP_W2;
    } else if (strcmp(s, "alpha") == 0) {
        c = MM3_LM_COMP_ALPHA;
    } else if (strcmp(s, "hada_w1_a") == 0) {
        c = MM3_LM_COMP_HADA_W1A;
    } else if (strcmp(s, "hada_w1_b") == 0) {
        c = MM3_LM_COMP_HADA_W1B;
    } else if (strcmp(s, "hada_w2_a") == 0) {
        c = MM3_LM_COMP_HADA_W2A;
    } else if (strcmp(s, "hada_w2_b") == 0) {
        c = MM3_LM_COMP_HADA_W2B;
    } else {
        return MM3_LM_COMP_NONE;
    }
    *layer  = (int) lyr;
    *module = mod;
    return c;
}

// One parse for both layouts. MM3_LM_COMP_NONE means "not an LM adapter key".
static MM3LmAdapterComp mm3_lm_adapter_parse_any(const std::string & key, int * layer, int * module) {
    MM3LmAdapterComp c = MM3_LM_COMP_NONE;
    if (mm3_lm_adapter_parse_key(key, layer, module, &c)) {
        return c;
    }
    return mm3_lm_adapter_parse_lokr(key, layer, module);
}

// Where a component lands in the pair. ALPHA is a scalar, not a stored tensor,
// so it has no slot and is handled during upload.
static ggml_tensor ** mm3_lm_pair_slot(MM3LmAdapterPair & p, MM3LmAdapterComp c) {
    switch (c) {
        case MM3_LM_COMP_A:   return &p.a;
        case MM3_LM_COMP_B:   return &p.b;
        case MM3_LM_COMP_W1:  return &p.w1;
        case MM3_LM_COMP_W2:  return &p.w2;
        case MM3_LM_COMP_W2A: return &p.w2_a;
        case MM3_LM_COMP_W2B: return &p.w2_b;
        case MM3_LM_COMP_MAG: return &p.m;
        // LoHa, mapped back onto our own naming: w?_a is a B, w?_b is an A.
        case MM3_LM_COMP_HADA_W1A: return &p.b;
        case MM3_LM_COMP_HADA_W1B: return &p.a;
        case MM3_LM_COMP_HADA_W2A: return &p.b2;
        case MM3_LM_COMP_HADA_W2B: return &p.a2;
        default:              return nullptr;
    }
}

// ─── adapter_config.json (2026-09-05) ───────────────────────────────────────
//
// `path` is the safetensors FILE; the config sits beside it. Everything here is
// optional and every field has a value that reproduces the pre-2026-09-05
// behaviour exactly, so an adapter written before this existed loads unchanged.
struct MM3LmAdapterCfg {
    float       ratio     = -1.0f;   // alpha/r, or alpha/sqrt(r) under rsLoRA
    bool        rslora    = false;
    bool        dora      = false;
    std::string peft_type = "LORA";
};

static MM3LmAdapterCfg mm3_lm_adapter_read_cfg(const std::string & sf_path) {
    MM3LmAdapterCfg out;
    const size_t    slash = sf_path.find_last_of("/\\");
    const std::string dir = (slash == std::string::npos) ? std::string(".") : sf_path.substr(0, slash);
    yyjson_doc *    doc   = yyjson_read_file((dir + "/adapter_config.json").c_str(), 0, NULL, NULL);
    if (!doc) {
        return out;
    }
    yyjson_val * root  = yyjson_doc_get_root(doc);
    double       alpha = 0.0, r = 0.0;
    if (root && yyjson_is_obj(root)) {
        yyjson_val * a  = yyjson_obj_get(root, "lora_alpha");
        yyjson_val * rv = yyjson_obj_get(root, "r");
        yyjson_val * rs = yyjson_obj_get(root, "use_rslora");
        yyjson_val * dv = yyjson_obj_get(root, "use_dora");
        yyjson_val * pt = yyjson_obj_get(root, "peft_type");
        if (a && yyjson_is_num(a)) alpha = yyjson_get_num(a);
        if (rv && yyjson_is_num(rv)) r = yyjson_get_num(rv);
        if (rs && yyjson_is_true(rs)) out.rslora = true;
        if (dv && yyjson_is_true(dv)) out.dora = true;
        if (pt && yyjson_is_str(pt)) out.peft_type = yyjson_get_str(pt);
    }
    yyjson_doc_free(doc);
    if (alpha > 0.0 && r > 0.0) {
        out.ratio = (float) (out.rslora ? alpha / sqrt(r) : alpha / r);
    }
    return out;
}

// Load a PEFT LM LoRA. Acquires its own backend reference (same shared pool
// as every other module). Returns nullptr with a message on any structural
// problem — a half-loaded adapter is worse than none (the LM-echo whitelist
// lesson: silently dropping modules changes what the adapter IS).
static MM3LmAdapter * mm3_lm_adapter_load(const char * path, std::string * err) {
    STFile st;
    if (!st_open(&st, path)) {
        if (err) {
            *err = std::string("cannot open adapter: ") + path;
        }
        return nullptr;
    }

    struct stat sb {};

    stat(path, &sb);

    MM3LmAdapter * ad = new MM3LmAdapter();
    ad->path          = path;
    ad->mtime         = (int64_t) sb.st_mtime;
    {
        const MM3LmAdapterCfg cfg = mm3_lm_adapter_read_cfg(path);
        ad->cfg_ratio  = cfg.ratio;
        ad->cfg_rslora = cfg.rslora;
        ad->cfg_dora   = cfg.dora;
        ad->cfg_peft   = cfg.peft_type;
        ad->is_hira    = cfg.peft_type == "HIRA";
    }
    ad->bp            = backend_init("MM3-LM-Adapter");
    ad->backend_ref   = true;
    ggml_backend_t backend = ad->bp.backend ? ad->bp.backend : ad->bp.cpu_backend;

    // Pass 1: count matched pairs so the ggml context can be sized exactly.
    int matched = 0;   // tensor-backed components only (alpha is a scalar)
    int n_alpha = 0;
    int n_mag   = 0;   // each magnitude also needs a same-shaped nrm tensor
    for (const STEntry & e : st.entries) {
        int                    layer, module;
        const MM3LmAdapterComp c = mm3_lm_adapter_parse_any(e.name, &layer, &module);
        if (c == MM3_LM_COMP_ALPHA) {
            n_alpha++;
        } else if (c == MM3_LM_COMP_MAG) {
            matched++;
            n_mag++;
        } else if (c != MM3_LM_COMP_NONE) {
            matched++;
        }
    }
    if (matched == 0) {
        st_close(&st);
        if (err) {
            *err = "no language_model LoRA/LoKr keys found (is this a DiT adapter?)";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }
    (void) n_alpha;

    ggml_init_params ip = {
        /*mem_size   =*/(size_t) (matched + n_mag + 4) * ggml_tensor_overhead(),
        /*mem_buffer =*/nullptr,
        /*no_alloc   =*/true,
    };
    ad->ctx = ggml_init(ip);

    // Pass 2: create tensors (f16, torch row-major maps directly onto ggml
    // [ne0 = innermost]): A [r, in] -> ggml [in, r]; B [out, r] -> ggml [r, out].
    for (const STEntry & e : st.entries) {
        int                    layer, module;
        const MM3LmAdapterComp c = mm3_lm_adapter_parse_any(e.name, &layer, &module);
        if (c == MM3_LM_COMP_NONE || c == MM3_LM_COMP_ALPHA) {
            continue;   // alpha is read during upload; it gets no tensor
        }
        if (c == MM3_LM_COMP_MAG) {
            // F32, not F16 like everything else here: nrm is computed in F32
            // and the two are divided elementwise in-graph, and a magnitude is
            // a per-output NORM whose values can sit well above the f16 range.
            int64_t n = 1;
            for (int d = 0; d < e.n_dims; d++) {
                n *= e.shape[d];
            }
            MM3LmAdapterPair & pm = ad->mods[layer][module];
            pm.m                  = ggml_new_tensor_1d(ad->ctx, GGML_TYPE_F32, n);
            pm.nrm                = ggml_new_tensor_1d(ad->ctx, GGML_TYPE_F32, n);
            ggml_set_name(pm.m, e.name.c_str());
            ad->dora_n++;
            continue;
        }
        if (e.n_dims != 2) {
            if (err) {
                *err = "unexpected adapter tensor rank on " + e.name;
            }
            st_close(&st);
            mm3_lm_adapter_free(ad);
            return nullptr;
        }
        // torch shape[0] = rows, shape[1] = cols; ggml ne0 is the innermost.
        // LoRA:  A [r, in]     -> [in, r]      B [out, r]   -> [r, out]
        // LoKr:  w1 [out_l, in_m] -> [in_m, out_l]; w2 [out_k, in_n] -> [in_n, out_k]
        //        w2_a [out_k, dim] -> [dim, out_k];  w2_b [dim, in_n] -> [in_n, dim]
        const int64_t ne0 = e.shape[1];
        const int64_t ne1 = e.shape[0];
        ggml_tensor * t   = ggml_new_tensor_2d(ad->ctx, GGML_TYPE_F16, ne0, ne1);
        ggml_set_name(t, e.name.c_str());
        MM3LmAdapterPair & p    = ad->mods[layer][module];
        ggml_tensor **     slot = mm3_lm_pair_slot(p, c);
        if (slot) {
            *slot = t;
        }
        if (c == MM3_LM_COMP_A) {
            ad->rank = (int) ne1;
        }
    }

    ad->buf = ggml_backend_alloc_ctx_tensors(ad->ctx, backend);
    if (!ad->buf) {
        st_close(&st);
        if (err) {
            *err = "adapter VRAM allocation failed";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }

    // Pass 3: upload, converting any dtype to f16 via f32.
    std::vector<float>      f32;
    std::vector<ggml_fp16_t> f16;
    for (const STEntry & e : st.entries) {
        int                    layer, module;
        const MM3LmAdapterComp c = mm3_lm_adapter_parse_any(e.name, &layer, &module);
        if (c == MM3_LM_COMP_NONE) {
            continue;
        }
        MM3LmAdapterPair & p = ad->mods[layer][module];
        if (c == MM3_LM_COMP_MAG) {
            const int64_t n = ggml_nelements(p.m);
            f32.resize((size_t) n);
            if (!adapter_to_f32(st_data(st, e), f32.data(), n, e.dtype)) {
                st_close(&st);
                if (err) {
                    *err = "unsupported dtype " + e.dtype + " on " + e.name;
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            ggml_backend_tensor_set(p.m, f32.data(), 0, (size_t) n * sizeof(float));
            continue;
        }
        if (c == MM3_LM_COMP_ALPHA) {
            // A scalar (or 1-element tensor). Stored, not uploaded.
            float av = 0.0f;
            if (adapter_to_f32(st_data(st, e), &av, 1, e.dtype)) {
                p.alpha_raw = av;
                p.has_alpha = true;
            }
            continue;
        }
        ggml_tensor ** slot = mm3_lm_pair_slot(p, c);
        ggml_tensor *  t    = slot ? *slot : nullptr;
        if (!t) {
            continue;
        }
        const int64_t n = ggml_nelements(t);
        f32.resize((size_t) n);
        if (!adapter_to_f32(st_data(st, e), f32.data(), n, e.dtype)) {
            st_close(&st);
            if (err) {
                *err = "unsupported dtype " + e.dtype + " on " + e.name;
            }
            mm3_lm_adapter_free(ad);
            return nullptr;
        }
        // THE F16 STORE IS LOSSY AT THE TOP OF THE RANGE, AND SILENTLY SO.
        //
        // Every checkpoint the MM3 LM trainer writes is BF16, whose exponent
        // reaches ~3.4e38. F16 stops at 65504. A factor above that converts to
        // +inf, and an inf in a LoKr factor turns the whole delta into inf, then
        // the residual add into NaN -- which is exactly the failure the AR loop
        // has been reporting as "N non-finite candidate logits were clamped to
        // -inf" (mm3-ar-loop.h) on training-preview renders. The generation then
        // completes "successfully" and returns noise.
        //
        // A checkpoint containing values that large has diverged; there is no
        // sensible rescue, so the load fails here with the tensor named rather
        // than a stderr line nobody reads. Non-finite input is caught the same
        // way: it means the trainer saved a broken step.
        {
            int64_t n_nonfinite = 0, n_overflow = 0;
            float   worst = 0.0f;
            for (int64_t i = 0; i < n; i++) {
                const float v = f32[(size_t) i];
                if (!std::isfinite(v)) {
                    n_nonfinite++;
                } else if (std::fabs(v) > 65504.0f) {
                    n_overflow++;
                    if (std::fabs(v) > worst) {
                        worst = std::fabs(v);
                    }
                }
            }
            if (n_nonfinite || n_overflow) {
                char buf[352];
                if (n_nonfinite) {
                    snprintf(buf, sizeof(buf),
                             "adapter tensor '%s' holds %lld non-finite values - this checkpoint diverged during "
                             "training and would generate noise",
                             e.name.c_str(), (long long) n_nonfinite);
                } else {
                    snprintf(buf, sizeof(buf),
                             "adapter tensor '%s' holds %lld values above the f16 limit (largest %.3g); they would "
                             "become +inf on load and turn every generation into noise",
                             e.name.c_str(), (long long) n_overflow, (double) worst);
                }
                st_close(&st);
                if (err) {
                    *err = buf;
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
        }

        f16.resize((size_t) n);
        ggml_fp32_to_fp16_row(f32.data(), f16.data(), n);
        ggml_backend_tensor_set(t, f16.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
    }
    st_close(&st);

    // Pass 4: validate pairing + count. Every module with an A must have a B.
    int n_lokr = 0;
    for (int l = 0; l < MM3_LM_ADAPTER_LAYERS; l++) {
        for (int m = 0; m < MM3_LM_ADAPTER_MODULES; m++) {
            MM3LmAdapterPair & p = ad->mods[l][m];
            if ((p.a == nullptr) != (p.b == nullptr)) {
                if (err) {
                    *err = "unpaired lora_A/lora_B at layer " + std::to_string(l);
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            // A LoKr module needs w1 and exactly one of {w2} or {w2_a, w2_b}.
            // Half a module is worse than none: it would silently apply a
            // different delta than the one that was trained.
            const bool any_lokr = p.w1 || p.w2 || p.w2_a || p.w2_b;
            if (any_lokr && !p.has_lokr()) {
                if (err) {
                    *err = "incomplete LoKr module at layer " + std::to_string(l) + " (need w1 plus w2 or w2_a/w2_b)";
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            if (p.has_lokr()) {
                // scale = alpha / dim. Monolithic w2 carries no dim, but
                // LyCORIS forces alpha == dim there, so the scale is exactly 1.
                // Factorized: w2_a is [dim, out_k] in ggml, so dim is ne[0].
                if (p.w2_a && p.has_alpha) {
                    const double dim = (double) p.w2_a->ne[0];
                    p.lokr_scale     = dim > 0.0 ? (float) (p.alpha_raw / dim) : 1.0f;
                } else {
                    p.lokr_scale = 1.0f;
                }
                // Geometry for the Kronecker apply, straight off the tensors.
                p.in_m  = p.w1->ne[0];
                p.out_l = p.w1->ne[1];
                p.in_n  = p.w2 ? p.w2->ne[0] : p.w2_b->ne[0];
                p.out_k = p.w2 ? p.w2->ne[1] : p.w2_a->ne[1];
                n_lokr++;
            }
            if (p.m && !p.a) {
                if (err) {
                    *err = "DoRA magnitude without a lora_A/lora_B pair at layer " + std::to_string(l);
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            if (p.m && p.m->ne[0] != p.b->ne[1]) {
                if (err) {
                    *err = "DoRA magnitude length does not match lora_B's output width at layer " +
                           std::to_string(l);
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            // Half a LoHa is worse than none: it would apply a plain LoRA
            // delta where a Hadamard was trained, silently.
            if ((p.a2 != nullptr) != (p.b2 != nullptr)) {
                if (err) {
                    *err = "unpaired LoHa hada_w2_a/hada_w2_b at layer " + std::to_string(l);
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            if (p.a2 && !p.a) {
                if (err) {
                    *err = "LoHa second pair without a first pair at layer " + std::to_string(l);
                }
                mm3_lm_adapter_free(ad);
                return nullptr;
            }
            if (p.has_loha()) {
                ad->is_loha = true;
            }
            if (p.a || p.has_lokr()) {
                ad->n_loaded++;
            }
        }
    }
    // use_dora and the tensors on disk must agree. Either alone means the file
    // was written by something that half-understood DoRA, and guessing which
    // half is right is how an adapter silently renders at the wrong strength.
    if (ad->cfg_dora != (ad->dora_n > 0)) {
        if (err) {
            *err = "adapter_config.json says use_dora=" + std::string(ad->cfg_dora ? "true" : "false") +
                   " but the file carries " + std::to_string(ad->dora_n) + " lora_magnitude_vector tensor(s)";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }
    // The norms need the resident base weights, which this loader does not
    // have. Until mm3_lm_dora_prepare() runs, nrm is UNINITIALISED and no graph
    // may read it.
    ad->dora_pending = ad->dora_n > 0;
    ad->is_lokr = n_lokr > 0;
    if (ad->is_lokr && ad->rank != 0) {
        if (err) {
            *err = "adapter mixes LoRA and LoKr modules";
        }
        mm3_lm_adapter_free(ad);
        return nullptr;
    }
    // Per-module base_scale. Precedence, and the compatibility rule that keeps
    // every already-shipped adapter byte-identical:
    //   1. a per-module `.alpha` scalar (comfy-style exports) -> alpha/rank;
    //   2. else adapter_config.json's alpha/r, or alpha/sqrt(r) under rsLoRA;
    //   3. else 1.0 — which is also what (2) computes for alpha == r without
    //      rsLoRA, so every SimpleTuner and every pre-2026-09-05 HOT-Step
    //      checkpoint lands on exactly 1.0, as it did before this read the file.
    float scale_shown = 1.0f;
    if (!ad->is_lokr) {
        for (int l = 0; l < MM3_LM_ADAPTER_LAYERS; l++) {
            for (int mo = 0; mo < MM3_LM_ADAPTER_MODULES; mo++) {
                MM3LmAdapterPair & p = ad->mods[l][mo];
                if (!p.has_lora()) {
                    continue;
                }
                const float rank_f = (float) p.a->ne[1];
                if (ad->cfg_rslora && ad->cfg_ratio > 0.0f) {
                    // rsLoRA cannot be recovered from a per-module alpha scalar
                    // (it is alpha/sqrt(r), and the scalar is just alpha), so
                    // the config wins whenever it says use_rslora.
                    p.base_scale = ad->cfg_ratio;
                } else if (p.has_alpha && rank_f > 0.0f) {
                    p.base_scale = p.alpha_raw / rank_f;
                } else if (ad->cfg_ratio > 0.0f) {
                    p.base_scale = ad->cfg_ratio;
                }
                scale_shown = p.base_scale;
            }
        }
    }
    fprintf(stderr, "[MM3] LM adapter loaded: %s (%d modules, %s%s%s%s%s, base scale %.4f, %.1f MB)\n", path, ad->n_loaded,
            ad->is_lokr ? "LoKr" : ad->is_loha ? "LoHa" : "LoRA", ad->is_hira ? " HiRA" : "",
            ad->cfg_rslora ? " +rsLoRA" : "", ad->dora_n ? " +DoRA" : "",
            (ad->is_loha || ad->is_hira) ? " (merge mode only)" : "",
            (double) scale_shown, (double) ggml_backend_buffer_get_size(ad->buf) / 1e6);
    return ad;
}
