#pragma once
// lm-hra.h — Householder Reflection Adaptation for the shared LM LoRA layer.
// HOT-Step file, 2026-09-05. Ported from train/dit-hra.h.
//
// Orthogonal fine-tuning without a Cayley transform: y = W (R x) with
// R = H_{r-1} ... H_0, H_k = I - 2 v_k v_k^T / ||v_k||^2. Each reflection is a
// rank-1 update on the ACTIVATIONS, so the whole thing is matmuls — the reason
// it is buildable in ggml where OFT/BOFT (a matrix inverse per step) are not.
// The vectors live in QwLoraPair::A as [in, r] and B stays null; column pairs
// start equal, so R = I and step 0 is exactly the frozen base. The forward is
// lm_hra_reflect in lm-graph.h.
//
// Export needs no loader change at all. R - I has rank <= r (each reflection
// only moves the span of its own vector), so with Q an orthonormal basis of
// span(v_0..v_{r-1})
//
//     W (R - I) = W (R - I) Q Q^T = (W U) Q^T,   U = (R - I) Q
//
// is an EXACT rank-r LoRA: A = Q, B = W U, alpha = r so the loader's alpha/r is
// 1. W U runs on the GPU through the trainer's scheduler (one [out, r] matmul
// per site, and mul_mat takes the base in its own dtype, so a q8_0 site needs
// no F32 copy). The raw vectors go to hra_vectors.safetensors beside the
// adapter, which is what a resume reads.

#include "qwen3-enc.h"
#include "train/lm-export.h"
#include "train/lm-graph.h"
#include "train/st-write.h"
#include "train/svd-host.h"

#include <string>
#include <vector>

// Holds the GPU scratch for the (W U) products across one export.
struct LmHraExport {
    const LmLora *        L     = nullptr;
    ggml_backend_sched_t  sched = nullptr;
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    ggml_tensor *         t_u   = nullptr;  // [in_max, r]
    ggml_tensor *         t_p   = nullptr;  // [out_max, r]
    std::vector<uint8_t>  arena;

    bool begin(const LmLora & bank, ggml_backend_sched_t sch, std::string * err) {
        L     = &bank;
        sched = sch;
        if (!bank.model || !sch) {
            *err = "HRA export: no model or scheduler handle";
            return false;
        }
        int64_t in_max = 0, out_max = 0;
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            ggml_tensor * w = lm_slot_weight(&bank.model->layers[bank.layer_lo], s);
            if (!w) {
                *err = "HRA export: a trained slot has no base weight";
                return false;
            }
            in_max  = std::max(in_max, w->ne[0]);
            out_max = std::max(out_max, w->ne[1]);
        }
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        ctx                 = ggml_init(ip);
        t_u                 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_max, bank.rank);
        t_p                 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_max, bank.rank);
        ggml_set_input(t_u);
        ggml_set_input(t_p);
        buf = ggml_backend_alloc_ctx_tensors(ctx, bank.model->backend);
        if (!buf) {
            *err = "HRA export: scratch allocation failed";
            ggml_free(ctx);
            ctx = nullptr;
            return false;
        }
        arena.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));
        return true;
    }

    void end() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        buf = nullptr;
        ctx = nullptr;
    }

    // One site's exported A = Q and B = W U. Returns false with *err empty when
    // the slot is not part of the bank.
    bool site(int l, int s, LmPeftFactors * f, std::string * err) {
        const QwLoraPair & pr = L->layers[l].p[s];
        if (!pr.A || !pr.hra) {
            err->clear();
            return false;
        }
        ggml_tensor * w   = lm_slot_weight(&L->model->layers[l], s);
        const int     r   = L->rank;
        const int     in  = (int) pr.A->ne[0];
        const int     out = (int) w->ne[1];

        std::vector<float> V((size_t) in * (size_t) r);
        ggml_backend_tensor_get(pr.A, V.data(), 0, V.size() * sizeof(float));

        // Q = orthonormal basis of span(v), U = (R - I) Q, both column-major.
        std::vector<double> Qd(V.begin(), V.end());
        hs_qr_mgs(Qd, in, r);
        std::vector<double> Ud(Qd);
        hs_householder_apply(V, in, r, Ud, r);
        for (size_t i = 0; i < Ud.size(); i++) {
            Ud[i] -= Qd[i];
        }

        std::vector<float> Uf(Ud.begin(), Ud.end());
        ggml_backend_tensor_set(t_u, Uf.data(), 0, Uf.size() * sizeof(float));
        {
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx2 = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx2, 64, false);
            ggml_tensor *    uv  = ggml_view_2d(ctx2, t_u, in, r, (size_t) in * sizeof(float), 0);
            // F32 on both sides: an F16 src0 would send U through cuBLAS's f16
            // conversion, and U's own values are what the exported B has to be
            // accurate in — a 1e-3 rounding there lands exactly on the gate's bar.
            ggml_tensor *    p   = ggml_mul_mat(ctx2, qwen3_f32(ctx2, w), uv);  // [out, r]
            ggml_build_forward_expand(
                gf, ggml_cpy(ctx2, p, ggml_view_2d(ctx2, t_p, out, r, (size_t) out * sizeof(float), 0)));
            ggml_backend_sched_reset(sched);
            const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
            ggml_free(ctx2);
            if (!ok) {
                *err = "HRA export: W U graph failed";
                return false;
            }
        }
        std::vector<float> P((size_t) out * (size_t) r);  // element (j, k) at k*out + j
        ggml_backend_tensor_get(t_p, P.data(), 0, P.size() * sizeof(float));

        f->in  = in;
        f->out = out;
        f->rr  = r;
        // A is ggml [in, r] = Q column-major, byte-identical to Qd cast to f32.
        f->A.resize((size_t) in * (size_t) r);
        for (size_t i = 0; i < f->A.size(); i++) {
            f->A[i] = (float) Qd[i];
        }
        // B is ggml [r, out]: element (k, j) at j*r + k.
        f->B.assign((size_t) r * (size_t) out, 0.0f);
        for (int j = 0; j < out; j++) {
            for (int k = 0; k < r; k++) {
                f->B[(size_t) j * (size_t) r + (size_t) k] = P[(size_t) k * (size_t) out + (size_t) j];
            }
        }
        return true;
    }
};

// hra_vectors.safetensors: the raw reflection vectors, which the exported
// rank-r LoRA does not contain (Q is a basis of their span, not the vectors).
// A resume reads this, exactly as the DiT does.
static bool lm_hra_write_vectors(const LmLora & L, const std::string & dir, std::string * err) {
    std::vector<STWTensor>          tensors;
    std::vector<std::vector<float>> store;
    for (int l = L.layer_lo; l < L.layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            const QwLoraPair & pr = L.layers[l].p[s];
            if (!pr.A || !pr.hra) {
                continue;
            }
            store.push_back(std::vector<float>((size_t) ggml_nelements(pr.A)));
            std::vector<float> & b = store.back();
            ggml_backend_tensor_get(pr.A, b.data(), 0, b.size() * sizeof(float));
            char nm[96];
            snprintf(nm, sizeof(nm), "L%d.s%d.hra_v", l, s);
            STWTensor st;
            st.name  = nm;
            st.shape = { pr.A->ne[1], pr.A->ne[0] };  // torch [r, in]
            st.data  = nullptr;
            tensors.push_back(st);
        }
    }
    for (size_t i = 0; i < tensors.size(); i++) {
        tensors[i].data = store[i].data();
    }
    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "format", "pt" });
    md.push_back({ "hot_step_hra_vectors", "v1" });
    const std::string vf = lm_join(dir, "hra_vectors.safetensors");
    if (!st_write_file(vf.c_str(), tensors, md, STW_F32)) {
        *err = "cannot write " + vf;
        return false;
    }
    return true;
}

// The whole HRA export: an exact rank-r PEFT LoRA plus the vector sidecar.
static bool lm_export_hra(const LmLora & L, const Qwen3LMConfig & cfg, const LmExportMeta & meta,
                          const std::string & out_dir, ggml_backend_sched_t sched, LmExportResult * res,
                          std::string * err, const LmExtraExport * extra = nullptr) {
    LmHraExport ex;
    if (!ex.begin(L, sched, err)) {
        return false;
    }
    LmPeftOverride ovr;
    ovr.rank  = L.rank;
    ovr.alpha = L.rank;  // scale 1: Q is orthonormal and B already carries W U
    ovr.site  = [&ex](int l, int s, LmPeftFactors * f, std::string * e) { return ex.site(l, s, f, e); };
    const bool ok = lm_export_peft(L, cfg, meta, out_dir, res, err, extra, &ovr);
    ex.end();
    if (!ok) {
        return false;
    }
    return lm_hra_write_vectors(L, out_dir, err);
}
