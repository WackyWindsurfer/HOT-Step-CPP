#pragma once
// qwen3-lora.h: minimal runtime-LoRA structs + graph helper for Qwen3 models.
//
// Kept deliberately tiny so qwen3-enc.h can include it without pulling in
// safetensors/json parsing.  The loader lives in lm-adapter.h.
//
// Runtime application (never merged into base weights, so it works on any
// base quantization):  y = W@x + B@(scale * (A@x))
//
// A: [H, r] (ggml ne0=H, ne1=r), B: [r, out] (ne0=r, ne1=out) — both loaded
// straight from PEFT lora_A/lora_B safetensors layouts with no transpose.
// scale = (lora_alpha / r) * user_scale, precomputed at load.
//
// Local HOT-Step feature — not upstream acestep.cpp.

#include "ggml.h"

#define QWEN3_LORA_MAX_LAYERS 64  // must cover QW3LM_MAX_LAYERS

enum QwLoraSlot {
    QW_LORA_Q = 0,
    QW_LORA_K,
    QW_LORA_V,
    QW_LORA_O,
    QW_LORA_GATE,
    QW_LORA_UP,
    QW_LORA_DOWN,
    QW_LORA_NSLOTS,
};

struct QwLoraPair {
    struct ggml_tensor * A     = nullptr;  // [H, r]
    struct ggml_tensor * B     = nullptr;  // [r, out]
    float                scale = 1.0f;     // (alpha/r) * user_scale

    // ─── LoKr (2026-07-30) ───────────────────────────────────────────────────
    //
    // Mutually exclusive with A/B: a site is either a LoRA or a LoKr, never
    // both. Populated by the TRAINER (train/lm-lokr.h) and, once it learns the
    // format, by the inference loader (lm-adapter.h) — which is the point of
    // hanging them here rather than on a trainer-private struct. One kron
    // implementation, one place to get the ggml indexing right, both paths.
    //
    // ggml layouts, matching what the DiT writes and adapter-merge.h reads:
    //   w1   [in_m, out_l]      w2   [in_n, out_k]        (monolithic)
    //   w2_a [dim,  out_k]      w2_b [in_n, dim]          (factorized)
    struct ggml_tensor * w1    = nullptr;
    struct ggml_tensor * w2    = nullptr;
    struct ggml_tensor * w2_a  = nullptr;
    struct ggml_tensor * w2_b  = nullptr;
    int64_t              in_m = 0, in_n = 0, out_l = 0, out_k = 0;
    float                lokr_scale = 1.0f;  // alpha / dim

    // ─── DoRA (Liu et al. 2024), 2026-09-05 ─────────────────────────────────
    //
    // W' = m * (W + s*BA) / ||W + s*BA||_col. `m` is the learned per-output
    // magnitude (a PARAM in the trainer, PEFT's lora_magnitude_vector on disk);
    // `nrm` holds the column norm of the CURRENT W + s*BA.
    //
    // The two runtimes differ in WHEN nrm is filled, not in what it means:
    //   * trainer   — refreshed once per optimizer window (lm_lora_dora_refresh),
    //                 because A/B move at every optimizer step;
    //   * inference — W, A and B are all frozen, so it is computed EXACTLY ONCE
    //                 at adapter load and reused for every token. That is what
    //                 makes DoRA correct in the unmerged runtime path here,
    //                 where the DiT's adapter-runtime.h only warns and defers to
    //                 merge mode.
    // Both null = not a DoRA adapter, and the emitted graph is unchanged.
    struct ggml_tensor * m   = nullptr;  // [out]
    struct ggml_tensor * nrm = nullptr;  // [out]

    // ─── LoHa (LyCORIS, Hyeon-Woo 2021) ─────────────────────────────────────
    // delta = (A1 B1) (.) (A2 B2). The second pair shares A/B's shapes. Both
    // null = not a LoHa. NOT applied by qwen3_linear_lora: the delta is a full
    // [in, out] tensor per module per forward, which at LM widths is a ~100 MB
    // transient per adapted projection PER TOKEN. The loaders refuse it in
    // runtime mode and point at merge mode, exactly as adapter-runtime.h does.
    struct ggml_tensor * A2 = nullptr;  // [in, r]
    struct ggml_tensor * B2 = nullptr;  // [r, out]

    // ─── HiRA (Huang et al., ICLR 2025) ─────────────────────────────────────
    // delta = W (.) (s*BA). Same A/B shapes as a LoRA, but the update is
    // modulated elementwise by the base, so it is NOT low-rank and has the same
    // runtime cost story as LoHa above — refused in runtime mode, merged.
    bool                 hira = false;

    // ─── PiSSA (Meng et al. 2024), 2026-09-05 ───────────────────────────────
    //
    // A/B start on the base weight's top-r singular directions. The DiT moves
    // that energy out of the base (W_res = W - s B0 A0) and trains against the
    // residual; the LM CANNOT — the shipped MM3 recipe trains against a q8_0
    // base, and there is nowhere to write an F32 residual back to. So the
    // residual is folded into the ADAPTER instead: the frozen base stays
    // untouched and the forward carries a constant negative copy of the init,
    //
    //   y = W x + s (B A - B0 A0) x,
    //
    // which is the same function (W_res + s B A) and the same zero delta at
    // step 0. A0/B0 are INPUTS, never params. Trainer-only: an exported PiSSA
    // run is a plain rank-2r LoRA (the two halves concatenated), so no loader
    // ever sees these.
    struct ggml_tensor * A0 = nullptr;  // [in, r] frozen
    struct ggml_tensor * B0 = nullptr;  // [r, out] frozen

    // ─── HRA (Yuan et al. 2024) ─────────────────────────────────────────────
    // y = W (R x), R = H_{r-1}...H_0, H_k = I - 2 v_k v_k^T/||v_k||^2, with the
    // vectors in A as [in, r] and B null. Trainer-only for the same reason
    // PiSSA is: W(R-I) has rank <= r, so the export is an exact plain rank-r
    // LoRA and the runtime needs to know nothing.
    bool                 hra = false;

    bool has_lokr() const { return w1 && (w2 || (w2_a && w2_b)); }
    bool has_dora() const { return m && nrm; }
    bool has_loha() const { return A2 && B2; }
    bool has_pissa() const { return A0 && B0; }
    // HOT-PiZZA (2026-09-06; named HOT-PiSSA for its first day — PiSSA with
    // principal-subspace dropout): the rank-dropout mask hits the trained principal
    // component B A but NOT the frozen -B0 A0 term, so under dropout the base's
    // own top-r subspace is stochastically deleted (and the rest scaled
    // 1/keep) every micro-step. Found by accident as a masking bug on
    // 2026-09-05; the adapter it produced beat every correctly-masked method
    // by ear on albumA twice, so it is kept on purpose behind this flag
    // (train/lm-graph.h says exactly what the forward is). Trainer-only.
    bool                 hot_pizza = false;
};

// LoKr delta: y += kron(w1, w2) . x, contracted factor-by-factor so the full
// [out, in] delta is never materialized.
//
// THE TOKEN AXIS MUST NOT REACH ne2 OF THESE MUL_MATS. Both contractions have a
// 2-D trainable factor as src0; if src1 carries the token count in ne2, ggml
// emits the weight gradient as out_prod(src1, grad) with dst->ne[2] == S, and
// ggml-cuda's out_prod takes its `dps2 > 1` fallback — one cublasSgemm PER
// TOKEN (out-prod.cu:96-108). On the DiT that made LoKr training 16.4x slower
// than the same run with a LoRA before it was found (2026-07-30).
//
// [in_n, in_m, S] and [in_n, in_m*S] are the SAME BYTES, so folding the token
// axis into the column count is a pure reshape and leaves ne2 == 1. The 3-D
// form is restored only around the permutes, which genuinely need it.
static inline struct ggml_tensor * qwen3_lokr_delta(struct ggml_context * ctx,
                                                    const QwLoraPair *    p,
                                                    struct ggml_tensor *  x,
                                                    struct ggml_tensor *  y) {
    struct ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    const int64_t        S  = ggml_nelements(xc) / xc->ne[0];

    struct ggml_tensor * X2 = ggml_reshape_2d(ctx, xc, p->in_n, p->in_m * S);
    struct ggml_tensor * T1 = p->w2 ? ggml_mul_mat(ctx, p->w2, X2)
                                    : ggml_mul_mat(ctx, p->w2_a, ggml_mul_mat(ctx, p->w2_b, X2));
    struct ggml_tensor * T13 = ggml_reshape_3d(ctx, T1, p->out_k, p->in_m, S);
    struct ggml_tensor * T1p = ggml_cont(ctx, ggml_permute(ctx, T13, 1, 0, 2, 3));
    struct ggml_tensor * T1f = ggml_reshape_2d(ctx, T1p, p->in_m, p->out_k * S);
    struct ggml_tensor * T2  = ggml_mul_mat(ctx, p->w1, T1f);
    struct ggml_tensor * T23 = ggml_reshape_3d(ctx, T2, p->out_l, p->out_k, S);
    struct ggml_tensor * T2p = ggml_cont(ctx, ggml_permute(ctx, T23, 1, 0, 2, 3));
    // THE SCALE GOES ON THE OUTPUT, not on w1. It is a scalar on a product, so
    // the two are identical maths (and identical gradients — d(s*w1*w2*x)/dw1 is
    // s*w2*x either way), but scaling w1 would force it to be F32: ggml_scale is
    // an F32 op, while mul_mat happily takes a BF16 src0. The inference loader
    // keeps the factors in their file dtype, so this is what lets a BF16 LoKr
    // load without a full F32 conversion of every factor.
    struct ggml_tensor * d = ggml_scale(ctx, T2p, p->lokr_scale);
    return ggml_add(ctx, y, ggml_reshape_4d(ctx, d, y->ne[0], y->ne[1], y->ne[2], y->ne[3]));
}

struct QwLoraLayer {
    QwLoraPair p[QW_LORA_NSLOTS];
};

// Set while loading a Qwen3 model whose layers will carry LoRA slots:
// disables QKV / gate-up weight fusion so each projection stays individually
// addressable (mirrors the DiT runtime-adapter fusion skip, dit.h:420).
inline bool g_qwen3_load_no_fuse = false;

// DoRA's per-output rescale: y <- y * (m / nrm). m/nrm is [out] and broadcasts
// over the token axis. Shared by every LoRA apply so the rescale cannot be
// wired into one path and silently forgotten in another.
static inline struct ggml_tensor * qwen3_lora_dora(struct ggml_context * ctx,
                                                   const QwLoraPair *    p,
                                                   struct ggml_tensor *  y) {
    if (!p || !p->has_dora()) {
        return y;
    }
    return ggml_mul(ctx, y, ggml_div(ctx, p->m, p->nrm));
}

// y = W@x (+ LoRA delta when the pair is populated).
static inline struct ggml_tensor * qwen3_linear_lora(struct ggml_context * ctx,
                                                     struct ggml_tensor *  w,
                                                     const QwLoraPair *    p,
                                                     struct ggml_tensor *  x) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    // Scale 0 is "adapter off", and off has to mean the BASE model — it is the
    // control arm of every A/B. The magnitude rescale is not zero-valued at
    // scale 0 (m is the trained magnitude and nrm was baked at the trained
    // scale, so m/nrm != 1), so it has to be skipped with the delta rather than
    // left applied to a base the delta never touched. lm_adapter_dora_prepare
    // folds user_scale into nrm, which is why this reads p->scale and not the
    // dial: at user_scale 0 the pair carries m/||W||, not 1.
    if (p && p->A && p->B && p->scale != 0.0f) {
        struct ggml_tensor * t = ggml_mul_mat(ctx, p->A, x);   // [r, S]
        t = ggml_scale(ctx, t, p->scale);                      // cheapest on the rank-r side
        y = ggml_add(ctx, y, ggml_mul_mat(ctx, p->B, t));      // [out, S]
        // DoRA. A no-op node-for-node when the pair carries no magnitude.
        y = qwen3_lora_dora(ctx, p, y);
    } else if (p && p->has_lokr()) {
        y = qwen3_lokr_delta(ctx, p, x, y);
    }
    return y;
}

// Convenience: fetch a slot pair from an optional per-layer slot table.
static inline const QwLoraPair * qwen3_lora_slot(const QwLoraLayer * ll, QwLoraSlot s) {
    return ll ? &ll->p[s] : nullptr;
}
