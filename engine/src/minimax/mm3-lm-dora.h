#pragma once
// minimax/mm3-lm-dora.h — the one-time DoRA norm pass for runtime-mode LM
// adapters. HOT-Step file (does not exist upstream), 2026-09-05.
//
// DoRA is  W' = m * (W + s*BA) / ||W + s*BA||_col.  `m` comes off disk with the
// adapter (PEFT's lora_magnitude_vector); the denominator does not, because it
// depends on the BASE WEIGHTS, which mm3_lm_adapter_load never sees.
//
// WHY THIS CAN BE A ONE-SHOT AT ALL. In the trainer the norm is refreshed every
// optimizer window because A and B move. At inference W, A and B are ALL
// FROZEN, so one pass is exact for every token of every generation. That is the
// difference that lets the MM3 runtime apply DoRA properly where the DiT's
// adapter-runtime.h only warns and defers to merge mode.
//
// WHAT THE DIALS DO NOT DO. The norm is computed at the adapter's own trained
// strength (`base_scale`), not at the request's global/attn/mlp/early-mid-late
// dials. Turning a dial still scales the delta — that is what the dial is for —
// but it does not re-derive the normaliser, so a dialled render is DoRA's
// direction at a different magnitude rather than a re-normalised DoRA. Baking
// the dials in would mean re-running this pass (a full [in,out] materialisation
// per module) on every scale change, and the dials exist precisely so that
// changing them is free.
//
// Cost: one [in, out] F32 transient per module (~200 MB at MM3's widths), a
// rank-r GEMM, and a row reduction — a few seconds at load, once per (adapter,
// resident base) pair. `MM3LmAdapter::dora_base` is what makes it once: a
// pristine LM reload changes the tensor addresses and the pass re-runs.

#include "mm3-lm-adapter.h"
#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>

// The base LM tensor an adapter module targets. Duplicated from
// mm3-lm-merge.h's mm3_lm_merge_target on purpose: this header is included by
// mm3-server.h, which must not pull in the merge path.
static ggml_tensor * mm3_lm_dora_target(const MM3Model & m, int layer, int module) {
    const MM3LmLayer & b = m.lm.blk[(size_t) layer];
    switch (module) {
        case MM3_LM_ADAPTER_Q:    return b.attn_q;
        case MM3_LM_ADAPTER_K:    return b.attn_k;
        case MM3_LM_ADAPTER_V:    return b.attn_v;
        case MM3_LM_ADAPTER_O:    return b.attn_output;
        case MM3_LM_ADAPTER_GATE: return b.ffn_gate;
        case MM3_LM_ADAPTER_UP:   return b.ffn_up;
        case MM3_LM_ADAPTER_DOWN: return b.ffn_down;
        default:                  return nullptr;
    }
}

// Fill every DoRA module's `nrm`. Idempotent: returns immediately when the
// adapter carries no magnitudes, or when this exact resident base was already
// normalised against.
static bool mm3_lm_dora_prepare(const MM3Model & m, MM3LmAdapter * ad, std::string * err) {
    if (!ad || ad->dora_n == 0) {
        return true;
    }
    const void * base_id = (const void *) m.lm.blk[0].attn_q;
    if (!ad->dora_pending && ad->dora_base == base_id) {
        return true;
    }
    if (!m.lm_resident) {
        if (err) {
            *err = "the LM is not resident; the DoRA norm pass must run after warm";
        }
        return false;
    }

    BackendPair          bp    = backend_init("MM3-LM-DoRA");
    ggml_backend_sched_t sched = backend_sched_new(bp, 64);
    bool                 ok    = true;
    int                  done  = 0;

    for (int l = 0; l < MM3_LM_ADAPTER_LAYERS && ok; l++) {
        for (int mod = 0; mod < MM3_LM_ADAPTER_MODULES && ok; mod++) {
            MM3LmAdapterPair & p = ad->mods[l][mod];
            if (!p.m || !p.nrm || !p.has_lora()) {
                continue;
            }
            ggml_tensor * w = mm3_lm_dora_target(m, l, mod);
            if (!w) {
                if (err) {
                    *err = "DoRA norm pass: missing base weight at layer " + std::to_string(l);
                }
                ok = false;
                break;
            }
            if (p.a->ne[0] != w->ne[0] || p.b->ne[1] != w->ne[1]) {
                if (err) {
                    *err = "DoRA norm pass: adapter/base shape mismatch at layer " + std::to_string(l);
                }
                ok = false;
                break;
            }
            // One small graph per module, exactly like the merge path's: the
            // [in, out] intermediate is the whole cost and holding 252 of them
            // at once is not on.
            const size_t         need = ggml_tensor_overhead() * 48 + ggml_graph_overhead_custom(64, false);
            std::vector<uint8_t> gvec(need);
            ggml_init_params     ip  = { need, gvec.data(), /*no_alloc*/ true };
            ggml_context *       ctx = ggml_init(ip);
            if (!ctx) {
                if (err) {
                    *err = "DoRA norm pass: ggml_init failed";
                }
                ok = false;
                break;
            }
            // ggml_cast dequantizes, which is the point: the resident base is
            // routinely q8_0 and that is why the runtime path exists at all.
            ggml_tensor * a_t   = ggml_cont(ctx, ggml_transpose(ctx, p.a));   // [r, in] f16
            ggml_tensor * b32   = ggml_cast(ctx, p.b, GGML_TYPE_F32);         // [r, out]
            ggml_tensor * delta = ggml_mul_mat(ctx, a_t, b32);                // [in, out] f32
            ggml_tensor * basef = ggml_cast(ctx, w, GGML_TYPE_F32);
            ggml_tensor * wd    = ggml_add(ctx, basef, ggml_scale(ctx, delta, p.base_scale));
            ggml_tensor * nr    = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, wd)));  // [1, out]
            ggml_tensor * outt  = ggml_reshape_1d(ctx, nr, nr->ne[1]);
            ggml_set_output(outt);

            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(gf, outt);
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, gf) ||
                ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
                if (err) {
                    *err = "DoRA norm pass: graph compute failed at layer " + std::to_string(l) + " (out of VRAM?)";
                }
                ggml_free(ctx);
                ok = false;
                break;
            }
            std::vector<float> host((size_t) ggml_nelements(outt));
            ggml_backend_tensor_get(outt, host.data(), 0, host.size() * sizeof(float));
            ggml_backend_tensor_set(p.nrm, host.data(), 0, host.size() * sizeof(float));
            ggml_free(ctx);
            done++;
        }
    }

    ggml_backend_sched_free(sched);
    backend_release(bp.backend, bp.cpu_backend);

    if (ok) {
        ad->dora_pending = false;
        ad->dora_base    = base_id;
        fprintf(stderr, "[MM3] LM adapter DoRA: ||W + s*BA||_col computed for %d module(s)\n", done);
    }
    return ok;
}
