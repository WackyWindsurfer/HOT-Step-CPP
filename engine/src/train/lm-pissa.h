#pragma once
// lm-pissa.h — PiSSA initialisation for the shared LM LoRA layer (Meng 2024).
// HOT-Step file, 2026-09-05. Ported from train/dit-pissa.h; the maths is the
// same, one design decision is not.
//
// A plain LoRA starts at A ~ N(0, 1/sqrt(in)), B = 0 and has to find the
// directions that matter. PiSSA starts the pair on the base weight's own top-r
// singular directions (A = V_r sqrt(S_r), B = sqrt(S_r) U_r^T) so the optimizer
// fine-tunes the principal subspace directly, which is where most of a weight's
// capacity lives. For that to leave the first forward unchanged, the same
// energy has to come OUT of the frozen path.
//
// THE ONE DEVIATION FROM THE DiT. dit_pissa_init writes W_res = W - s B0 A0
// back into the F32 mirror. The LM cannot: mm3-lm-train's shipped recipe trains
// against a q8_0 base (the f16 one leaves no room for the optimizer), and there
// is no F32 mirror to receive a residual — requantizing W_res would perturb
// every weight in the model to fix r directions. So the residual is folded into
// the ADAPTER instead. lm_lora_init allocates a frozen A0/B0 per site and
// lm_linear emits
//
//     y = W x + s (B A) x - s (B0 A0) x,
//
// which is exactly (W - s B0 A0) + s B A as a function, starts at exactly W,
// and never touches a base weight. The cost is two extra rank-r mul_mats per
// site per forward and 2x the adapter's parameter memory (A0/B0 are the same
// shape as A/B) — paid only under --pissa.
//
// The SVD is randomized, with the two big products on the GPU:
//   stage T (GPU)  W^T into a scratch, plus ||W||_F^2   — one cast per site,
//                  which is also how a q8_0 site gets an F32 view of itself
//   stage A (GPU)  Y = (W W^T)^iters W Omega            q = r + oversample
//   stage B (host) Q = qr(Y)                            modified Gram-Schmidt
//   stage C (GPU)  C = Q^T W                            [q, in]
//   stage D (host) C C^T = Uc diag(lambda) Uc^T (Jacobi), S = sqrt(lambda),
//                  U_r = Q Uc_r, V_r = C^T Uc_r / S_r
//
// The captured-energy line (sum lambda_k / ||W||_F^2) is the correctness
// diagnostic: a working SVD captures far more than a random rank-r subspace's
// r/min(in,out).
//
// Export (lm-hra.h's neighbour, in lm-export.h) is a plain PEFT LoRA of rank 2r
// on the ORIGINAL base — s(BA - B0A0) = s [B, -B0] [A; A0] — so the merge and
// runtime paths need no PiSSA knowledge whatsoever.

#include "backend.h"
#include "qwen3-enc.h"
#include "train/lm-export.h"
#include "train/lm-graph.h"
#include "train/svd-host.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct LmPissaStats {
    double energy_min = 1.0, energy_mean = 0.0;  // captured / ||W||_F^2 over sites
    int    sites      = 0;
};

// Fills A/B with the SVD factors and A0/B0 with a frozen copy of them. The base
// is not written. `sched` runs the GPU stages; it is reset around every graph,
// so call this before the training graphs exist.
static bool lm_pissa_init(LmLora * L, ggml_backend_sched_t sched, int oversample, int iters, LmPissaStats * stats,
                          std::string * err) {
    Qwen3LM * lm = L->model;
    if (!lm || !L->pissa || L->layer_lo >= L->layer_hi) {
        *err = "PiSSA: adapter not initialised for it";
        return false;
    }
    const int r = L->rank;
    const int q = r + std::max(0, oversample);
    iters       = std::max(0, std::min(4, iters));

    // Scratch sized by the largest site in the trained window.
    int64_t in_max = 0, out_max = 0;
    for (int s = 0; s < QW_LORA_NSLOTS; s++) {
        ggml_tensor * w = lm_slot_weight(&lm->layers[L->layer_lo], s);
        if (!w) {
            *err = "PiSSA: a trained slot has no base weight (fused projections?)";
            return false;
        }
        in_max  = std::max(in_max, w->ne[0]);
        out_max = std::max(out_max, w->ne[1]);
    }
    if ((int64_t) q > std::min(in_max, out_max)) {
        *err = "PiSSA: rank + oversample exceeds the smallest weight dimension";
        return false;
    }

    ggml_context * sctx = nullptr;
    {
        ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        sctx                = ggml_init(ip);
    }
    // t_wt is the site's weight TRANSPOSED and in F32: [out, in]. Both stages
    // that contract over the output axis need it, and building it with
    // qwen3_f32 is what gives a q8_0 site an F32 view of itself without
    // dequantizing the whole model.
    ggml_tensor * t_wt    = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, in_max);
    ggml_tensor * t_omega = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, in_max, q);
    ggml_tensor * t_y     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, q);
    ggml_tensor * t_q     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, q);
    ggml_tensor * t_c     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, q, in_max);
    ggml_tensor * t_en    = ggml_new_tensor_1d(sctx, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { t_wt, t_omega, t_y, t_q, t_c, t_en }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, lm->backend);
    if (!sbuf) {
        *err = "PiSSA: scratch allocation failed";
        ggml_free(sctx);
        return false;
    }
    fprintf(stderr, "[pissa] scratch %.1f MB (largest site %lld x %lld), rank %d + oversample %d, %d power iteration(s)\n",
            (double) ggml_backend_buffer_get_size(sbuf) / 1048576.0, (long long) in_max, (long long) out_max, r,
            std::max(0, oversample), iters);
    auto done = [&](bool ok) {
        ggml_backend_buffer_free(sbuf);
        ggml_free(sctx);
        return ok;
    };

    std::vector<uint8_t> arena(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false));
    auto                 compute = [&](ggml_cgraph * gf, ggml_context * ctx) -> bool {
        ggml_backend_sched_reset(sched);
        const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
        return ok;
    };
    auto view2 = [&](ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1) {
        return ggml_view_2d(ctx, t, ne0, ne1, (size_t) ne0 * sizeof(float), 0);
    };

    // The factors are stored so that s * B0 A0 reproduces W's rank-r truncation
    // EXACTLY, whatever the scale is. The DiT bakes sqrt(S) into each factor and
    // then subtracts s*B0A0, which only lands on the true PiSSA residual when
    // alpha == rank; dividing both factors by sqrt(s) here makes it right for
    // any alpha and for rsLoRA, at zero cost when s == 1.
    const double inv_sqrt_s = (L->scale > 0.0f) ? 1.0 / sqrt((double) L->scale) : 1.0;

    LmRng rng;
    lm_rng_seed(&rng, 0x9155a000ull ^ (uint64_t) L->rank);
    std::vector<float> host;
    double             e_sum = 0.0;
    int                n_e   = 0;

    for (int l = L->layer_lo; l < L->layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair &  pr = L->layers[l].p[s];
            if (!pr.A || !pr.B || !pr.A0 || !pr.B0) {
                continue;
            }
            ggml_tensor * w   = lm_slot_weight(&lm->layers[l], s);
            if (!w) {
                *err = "PiSSA: a trained slot has no base weight";
                return done(false);
            }
            const int in  = (int) w->ne[0];
            const int out = (int) w->ne[1];

            // ── T: W^T (F32) into the scratch, and ||W||_F^2 ───────────────
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                ggml_tensor *    wf  = qwen3_f32(ctx, w);                          // [in, out]
                ggml_tensor *    wt  = ggml_cont(ctx, ggml_transpose(ctx, wf));    // [out, in]
                ggml_build_forward_expand(gf, ggml_cpy(ctx, wt, view2(ctx, t_wt, out, in)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_sum(ctx, ggml_sqr(ctx, wf)), t_en));
                if (!compute(gf, ctx)) {
                    *err = "PiSSA: transpose/norm graph failed";
                    return done(false);
                }
            }
            float w_fro2 = 0.0f;
            ggml_backend_tensor_get(t_en, &w_fro2, 0, sizeof(float));

            // ── A: Y = (W W^T)^iters W Omega ──────────────────────────────
            host.assign((size_t) in * (size_t) q, 0.0f);
            lm_rng_fill_normal(&rng, host, 1.0f);
            ggml_backend_tensor_set(t_omega, host.data(), 0, host.size() * sizeof(float));
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                // BOTH operands are F32 here, and that is not tidiness. Using
                // the base in its own F16 dtype sends src1 through cuBLAS's F16
                // conversion, and the power iteration deliberately grows Y — it
                // reached 2.5e6 on layer 0's q_proj — so the SECOND iteration's
                // src1 goes over f16's 65504 and the whole SVD comes back NaN.
                // It surfaced as a dead depth-loss gradient forty lines later.
                ggml_tensor * wf  = qwen3_f32(ctx, w);
                ggml_tensor * wtv = view2(ctx, t_wt, out, in);
                ggml_tensor * y   = ggml_mul_mat(ctx, wf, view2(ctx, t_omega, in, q));  // [out, q]
                for (int it = 0; it < iters; it++) {
                    ggml_tensor * z = ggml_mul_mat(ctx, wtv, y);                        // [in, q]
                    y               = ggml_mul_mat(ctx, wf, z);                         // [out, q]
                }
                ggml_build_forward_expand(gf, ggml_cpy(ctx, y, view2(ctx, t_y, out, q)));
                if (!compute(gf, ctx)) {
                    *err = "PiSSA: stage A graph failed";
                    return done(false);
                }
            }
            // ── B: Q = qr(Y) on the host ──────────────────────────────────
            std::vector<double> Yd((size_t) out * (size_t) q);
            host.resize(Yd.size());
            ggml_backend_tensor_get(t_y, host.data(), 0, host.size() * sizeof(float));
            double ymax = 0.0;
            for (size_t i = 0; i < Yd.size(); i++) {
                Yd[i] = (double) host[i];
                ymax  = std::max(ymax, std::fabs(Yd[i]));
            }
            if (!std::isfinite(ymax)) {
                char b[224];
                snprintf(b, sizeof(b), "PiSSA: layer %d slot %s: the range-finder product is non-finite", l,
                         lm_slot_peft_name(s));
                *err = b;
                return done(false);
            }
            hs_qr_mgs(Yd, out, q);
            for (size_t i = 0; i < Yd.size(); i++) {
                host[i] = (float) Yd[i];
            }
            ggml_backend_tensor_set(t_q, host.data(), 0, host.size() * sizeof(float));
            // ── C: C = Q^T W  [q, in] ─────────────────────────────────────
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                ggml_tensor *    c   = ggml_mul_mat(ctx, view2(ctx, t_q, out, q), view2(ctx, t_wt, out, in));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, c, view2(ctx, t_c, q, in)));
                if (!compute(gf, ctx)) {
                    *err = "PiSSA: stage C graph failed";
                    return done(false);
                }
            }
            // ── D: small SVD on the host, then A0/B0 ──────────────────────
            std::vector<double> Cd((size_t) q * (size_t) in);  // element (k, i) at i*q + k
            host.resize(Cd.size());
            ggml_backend_tensor_get(t_c, host.data(), 0, host.size() * sizeof(float));
            double cmax = 0.0;
            for (size_t i = 0; i < Cd.size(); i++) {
                Cd[i] = (double) host[i];
                cmax  = std::max(cmax, std::fabs(Cd[i]));
            }
            if (!std::isfinite(cmax)) {
                char b[224];
                snprintf(b, sizeof(b), "PiSSA: layer %d slot %s: Q^T W is non-finite (|Y| max %.4g)", l,
                         lm_slot_peft_name(s), ymax);
                *err = b;
                return done(false);
            }
            std::vector<double> G((size_t) q * (size_t) q, 0.0);
            for (int i = 0; i < in; i++) {
                const double * ci = Cd.data() + (size_t) i * (size_t) q;
                for (int k = 0; k < q; k++) {
                    for (int l2 = k; l2 < q; l2++) {
                        G[(size_t) k * (size_t) q + (size_t) l2] += ci[k] * ci[l2];
                    }
                }
            }
            for (int k = 0; k < q; k++) {
                for (int l2 = 0; l2 < k; l2++) {
                    G[(size_t) k * (size_t) q + (size_t) l2] = G[(size_t) l2 * (size_t) q + (size_t) k];
                }
            }
            std::vector<double> evals, evecs;
            hs_jacobi_sym(G, q, evals, evecs);
            std::vector<int> order((size_t) q);
            for (int k = 0; k < q; k++) {
                order[(size_t) k] = k;
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) { return evals[(size_t) a] > evals[(size_t) b]; });

            std::vector<float> A0((size_t) in * (size_t) r), B0((size_t) r * (size_t) out);
            double             captured = 0.0;
            for (int k = 0; k < r; k++) {
                const int      idx = order[(size_t) k];
                const double   lam = std::max(evals[(size_t) idx], 0.0);
                const double   S   = sqrt(lam);
                // Both factors carry sqrt(S_k)/sqrt(s), so B0 A0 = W_topr / s
                // and the graph's s * B0 A0 is W's rank-r truncation exactly.
                const double   sS  = sqrt(S) * inv_sqrt_s;
                const double * uc  = evecs.data() + (size_t) idx * (size_t) q;  // Uc[:, idx]
                captured += lam;
                // V_r[i] = C^T uc / S ; A0[i, k] = V_r[i] * sqrt(S)  (ggml A: [in, r], (i,k) at k*in+i)
                for (int i = 0; i < in; i++) {
                    const double * ci = Cd.data() + (size_t) i * (size_t) q;
                    double         v  = 0.0;
                    for (int kk = 0; kk < q; kk++) {
                        v += ci[kk] * uc[kk];
                    }
                    A0[(size_t) k * (size_t) in + (size_t) i] = (S > 0.0) ? (float) (v / S * sS) : 0.0f;
                }
                // U_r[j] = Q uc ; B0[k, j] = sqrt(S) * U_r[j]        (ggml B: [r, out], (k,j) at j*r+k)
                for (int j = 0; j < out; j++) {
                    double u = 0.0;
                    for (int kk = 0; kk < q; kk++) {
                        u += Yd[(size_t) kk * (size_t) out + (size_t) j] * uc[kk];
                    }
                    B0[(size_t) j * (size_t) r + (size_t) k] = (float) (sS * u);
                }
            }
            ggml_backend_tensor_set(pr.A, A0.data(), 0, A0.size() * sizeof(float));
            ggml_backend_tensor_set(pr.B, B0.data(), 0, B0.size() * sizeof(float));
            ggml_backend_tensor_set(pr.A0, A0.data(), 0, A0.size() * sizeof(float));
            ggml_backend_tensor_set(pr.B0, B0.data(), 0, B0.size() * sizeof(float));

            // A NaN here would silently poison the adapter — A0/B0 are frozen
            // inputs, so the very first forward would be non-finite and the run
            // would die somewhere far from the cause. Refuse at the site.
            if (!std::isfinite(captured) || !std::isfinite((double) w_fro2)) {
                char b[288];
                snprintf(b, sizeof(b),
                         "PiSSA: layer %d slot %s produced a non-finite SVD (||W||_F^2 %.6g, captured %.6g, "
                         "top eigenvalue %.6g, base dtype %s, %dx%d)",
                         l, lm_slot_peft_name(s), (double) w_fro2, captured, evals[(size_t) order[0]],
                         ggml_type_name(w->type), in, out);
                *err = b;
                return done(false);
            }
            const double frac = (w_fro2 > 0.0f) ? captured / (double) w_fro2 : 0.0;
            if (stats) {
                stats->energy_min = std::min(stats->energy_min, frac);
                stats->sites++;
            }
            e_sum += frac;
            n_e++;
        }
    }
    if (stats && n_e > 0) {
        stats->energy_mean = e_sum / (double) n_e;
    }
    return done(true);
}

// Both LM trainers size their training scheduler from the training graph, which
// is built long after the adapter bank exists — so the init gets one of its
// own, and the ~200 MB of scratch it holds is returned before anything else is
// allocated.
static bool lm_pissa_init_standalone(LmLora * L, int oversample, int iters, LmPissaStats * stats, std::string * err) {
    if (!L->model) {
        *err = "PiSSA: the adapter bank is not attached to a model";
        return false;
    }
    BackendPair bp;
    bp.backend     = L->model->backend;
    bp.cpu_backend = L->model->cpu_backend;
    bp.has_gpu     = bp.backend != bp.cpu_backend;
    ggml_backend_sched_t sc = backend_sched_new(bp, 2048);
    const bool           ok = lm_pissa_init(L, sc, oversample, iters, stats, err);
    ggml_backend_sched_free(sc);
    return ok;
}

// The PiSSA export: a plain rank-2r LoRA on the ORIGINAL base.
//
// TWO CHOICES HERE, both about numbers rather than algebra.
//
// 1. THE FACTORIZATION. The obvious one is the DiT's,
//        s(BA - B0A0) = s [B, -B0] [A; A0],
//    and it is a catastrophic-cancellation trap: |B| and |B0| are both the full
//    SVD magnitude while their difference is the whole adapter, so two nearly
//    equal halves have to cancel to something ~1% of themselves. The MM3 runtime
//    loader stores adapter tensors as f16 (~5e-4 per element), which is applied
//    to the HALVES — the surviving error is then tens of percent OF THE DELTA.
//    Measured at 2.1e-3 against a delta that was itself ~0, i.e. unbounded.
//    The identity
//        BA - B0A0 = (B - B0) A + B0 (A - A0)
//    is exactly as rank-2r, but every term is now the size of the DRIFT rather
//    than the size of the base's singular values, and the two ADD instead of
//    cancelling. Rounding is then 1e-3 of the delta, which is what it should be.
//
// 2. THE SCALE. Baked into the first factor rather than carried as a doubled
//    alpha (the DiT's convention), because alpha is an integer in the config and
//    s = alpha/sqrt(r) under rsLoRA is not. Baking keeps one code path exact for
//    every scale and reduces to a scale of 1 at the default alpha == rank.
static bool lm_export_pissa(const LmLora & L, const Qwen3LMConfig & cfg, const LmExportMeta & meta,
                            const std::string & out_dir, LmExportResult * res, std::string * err,
                            const LmExtraExport * extra = nullptr) {
    const int          r = L.rank;
    std::vector<float> a, b, a0, b0;
    LmPeftOverride     ovr;
    ovr.rank  = 2 * r;
    ovr.alpha = 2 * r;  // loader scale alpha/rank = 1
    ovr.site  = [&](int l, int s, LmPeftFactors * f, std::string * e) -> bool {
        const QwLoraPair & pr = L.layers[l].p[s];
        if (!pr.A || !pr.B || !pr.has_pissa()) {
            e->clear();
            return false;
        }
        const int64_t in  = pr.A->ne[0];
        const int64_t out = pr.B->ne[1];
        a.assign((size_t) in * (size_t) r, 0.0f);
        b.assign((size_t) r * (size_t) out, 0.0f);
        a0.assign(a.size(), 0.0f);
        b0.assign(b.size(), 0.0f);
        ggml_backend_tensor_get(pr.A, a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_get(pr.B, b.data(), 0, b.size() * sizeof(float));
        ggml_backend_tensor_get(pr.A0, a0.data(), 0, a0.size() * sizeof(float));
        ggml_backend_tensor_get(pr.B0, b0.data(), 0, b0.size() * sizeof(float));

        f->in  = in;
        f->out = out;
        f->rr  = 2 * r;
        // A_cat is ggml [in, 2r]: columns 0..r-1 are A, r..2r-1 are (A - A0).
        f->A.resize(a.size() + a0.size());
        std::copy(a.begin(), a.end(), f->A.begin());
        for (size_t i = 0; i < a0.size(); i++) {
            f->A[a.size() + i] = a[i] - a0[i];
        }
        // B_cat is ggml [2r, out]: per output column j, s * [(B - B0)(j), B0(j)].
        f->B.assign((size_t) 2 * (size_t) r * (size_t) out, 0.0f);
        for (int64_t j = 0; j < out; j++) {
            for (int k = 0; k < r; k++) {
                const size_t src = (size_t) j * (size_t) r + (size_t) k;
                f->B[(size_t) j * (size_t) (2 * r) + (size_t) k] = L.scale * (b[src] - b0[src]);
                f->B[(size_t) j * (size_t) (2 * r) + (size_t) r + (size_t) k] = L.scale * b0[src];
            }
        }
        return true;
    };
    return lm_export_peft(L, cfg, meta, out_dir, res, err, extra, &ovr);
}
