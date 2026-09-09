#pragma once
// train/mm3-lm-verify-export.h — export -> load round trip for MM3 LM adapters.
// HOT-Step file, 2026-09-05.
//
// WHY. A trainer can export a perfectly good file that the runtime then loads
// as something else: the turbo8 no-op in reverse. rsLoRA is the sharpest case —
// alpha/sqrt(r) is 16x alpha/r at r256 — but DoRA's magnitudes and LoHa's
// hada_w* keys have the same shape of failure, where every gate passes and the
// adapter renders at the wrong strength (or does nothing).
//
// So this runs THE ACTUAL RUNTIME LOADER (minimax/mm3-lm-adapter.h) over the
// directory the trainer just wrote, and checks three things per site:
//
//   1. the loader's base_scale equals the scale the training graph applied;
//   2. the tensors it staged equal the trainer's, within f16 rounding (the
//      runtime stores f16 on purpose — the trainer's are F32);
//   3. the loader recognised the parameterization (DoRA magnitudes present and
//      the right length; LoHa's second pair present; HiRA flagged).
//
// It is a check, not a conversion: nothing here writes.

#include "minimax/mm3-lm-adapter.h"
#include "train/lm-graph.h"
#include "train/svd-host.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ─── PiSSA / HRA: the delta gate ────────────────────────────────────────────
//
// These two do NOT export their trainer tensors, so the tensor-for-tensor
// comparison above is meaningless for them: PiSSA writes [B, -B0][A; A0] at
// rank 2r with the scale baked into B, and HRA writes (W U) Q^T, which shares
// not one number with the reflection vectors it came from. What CAN be checked
// is the only thing that matters — that the exported adapter applies the SAME
// LINEAR MAP the training graph applied.
//
// So: probe both with random vectors, once per site, and compare
//
//   loaded   base_scale * B_l^T (A_l^T x)          (through the runtime loader)
//   PiSSA    s * (B^T (A^T x) - B0^T (A0^T x))     (the folded residual form)
//   HRA      W (R x) - W x                          (the reflections, on the
//                                                    dequantized base)
//
// The HRA arm is the one that needs the F32 base: (W U) Q^T is only equal to
// W(R - I) if Q really is an orthonormal basis of span(v), and nothing short of
// multiplying by W says so.
static bool mm3_lm_verify_export_delta(const std::string & dir, const LmLora & L, std::string * err,
                                       const std::string & base_lm_path = "") {
    const std::string sf = dir + "/adapter_model.safetensors";
    std::string       lerr;
    // The base path is how the loader finds a delta-form adapter's residual.
    MM3LmAdapter *    ad = mm3_lm_adapter_load(sf.c_str(), &lerr, base_lm_path.empty() ? nullptr : base_lm_path.c_str());
    if (!ad) {
        *err = "the runtime loader refused the file we just wrote: " + lerr;
        return false;
    }
    auto pull = [&](ggml_tensor * t, std::vector<float> * v) {
        v->assign((size_t) ggml_nelements(t), 0.0f);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, v->data(), 0, v->size() * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(h.data(), v->data(), (int64_t) h.size());
        } else {
            v->clear();
        }
    };

    // Two sites: one square (q_proj) and one rectangular (k_proj), so a layout
    // bug that happens to be symmetric cannot hide.
    const int  slots[2] = { QW_LORA_Q, QW_LORA_K };
    const int  layer    = L.layer_lo;
    const int  NPROBE   = 8;
    LmRng      rng;
    lm_rng_seed(&rng, 0xBADC0FFEull);

    int    bad = 0, checked = 0;
    double worst = 0.0;
    // The runtime loader stores adapter tensors as f16 — 2^-11 = 4.9e-4 per
    // element — and a rank-2r reconstruction compounds two of them, so the floor
    // here is a few times that whatever the trainer does. 3e-3 is the honest
    // bar; anything materially above it means the bytes changed meaning rather
    // than precision, which is the failure this gate exists to catch.
    const double BAR = 3e-3;

    for (int si = 0; si < 2; si++) {
        const int          s  = slots[si];
        const QwLoraPair & q  = L.layers[layer].p[s];
        const MM3LmAdapterPair & p = ad->mods[layer][s];
        if (!q.A || !p.has_lora()) {
            fprintf(stderr, "[verify-delta] layer %d slot %s: nothing to compare\n", layer, lm_slot_peft_name(s));
            bad++;
            continue;
        }
        ggml_tensor * w   = lm_slot_weight(&L.model->layers[layer], s);
        const int64_t in  = q.A->ne[0];
        const int64_t out = w->ne[1];
        const int64_t rr  = p.a->ne[1];

        std::vector<float> Al, Bl;
        pull(p.a, &Al);
        pull(p.b, &Bl);
        if ((int64_t) Al.size() != in * rr || (int64_t) Bl.size() != rr * out) {
            fprintf(stderr, "[verify-delta] layer %d slot %s: loaded shapes %zu / %zu do not match [%lld,%lld]\n",
                    layer, lm_slot_peft_name(s), Al.size(), Bl.size(), (long long) in, (long long) out);
            bad++;
            continue;
        }

        // Trainer-side factors (PiSSA) or the base weight (HRA).
        std::vector<float> A, B, A0, B0, Wh, V;
        if (L.pissa) {
            A.assign((size_t) ggml_nelements(q.A), 0.0f);
            B.assign((size_t) ggml_nelements(q.B), 0.0f);
            A0.assign(A.size(), 0.0f);
            B0.assign(B.size(), 0.0f);
            ggml_backend_tensor_get(q.A, A.data(), 0, A.size() * sizeof(float));
            ggml_backend_tensor_get(q.B, B.data(), 0, B.size() * sizeof(float));
            ggml_backend_tensor_get(q.A0, A0.data(), 0, A0.size() * sizeof(float));
            ggml_backend_tensor_get(q.B0, B0.data(), 0, B0.size() * sizeof(float));
        } else {
            V.assign((size_t) ggml_nelements(q.A), 0.0f);
            ggml_backend_tensor_get(q.A, V.data(), 0, V.size() * sizeof(float));
            std::string derr;
            if (!lm_host_dequant(w, &Wh, &derr)) {
                *err = derr;
                mm3_lm_adapter_free(ad);
                return false;
            }
        }

        double site_worst = 0.0, ref_mag = 0.0;
        std::vector<float>  x((size_t) in);
        std::vector<double> ref((size_t) out), got((size_t) out);
        for (int t = 0; t < NPROBE; t++) {
            lm_rng_fill_normal(&rng, x, 1.0f);

            // Loaded: base_scale * B_l^T (A_l^T x).
            std::vector<double> tl((size_t) rr, 0.0);
            for (int64_t k = 0; k < rr; k++) {
                double acc = 0.0;
                for (int64_t i = 0; i < in; i++) {
                    acc += (double) Al[(size_t) (k * in + i)] * (double) x[(size_t) i];
                }
                tl[(size_t) k] = acc;
            }
            for (int64_t j = 0; j < out; j++) {
                double acc = 0.0;
                for (int64_t k = 0; k < rr; k++) {
                    acc += (double) Bl[(size_t) (j * rr + k)] * tl[(size_t) k];
                }
                got[(size_t) j] = (double) p.base_scale * acc;
            }

            // Reference.
            if (L.pissa) {
                const int64_t       r = q.A->ne[1];
                std::vector<double> t1((size_t) r, 0.0), t0((size_t) r, 0.0);
                for (int64_t k = 0; k < r; k++) {
                    double a1 = 0.0, a0 = 0.0;
                    for (int64_t i = 0; i < in; i++) {
                        a1 += (double) A[(size_t) (k * in + i)] * (double) x[(size_t) i];
                        a0 += (double) A0[(size_t) (k * in + i)] * (double) x[(size_t) i];
                    }
                    t1[(size_t) k] = a1;
                    t0[(size_t) k] = a0;
                }
                for (int64_t j = 0; j < out; j++) {
                    double acc = 0.0;
                    for (int64_t k = 0; k < r; k++) {
                        acc += (double) B[(size_t) (j * r + k)] * t1[(size_t) k] -
                               (double) B0[(size_t) (j * r + k)] * t0[(size_t) k];
                    }
                    ref[(size_t) j] = (double) L.scale * acc;
                }
            } else {
                const int           r = L.rank;
                std::vector<double> xr(x.begin(), x.end());
                hs_householder_apply(V, (int) in, r, xr, 1);
                for (int64_t i = 0; i < in; i++) {
                    xr[(size_t) i] -= (double) x[(size_t) i];   // (R - I) x
                }
                for (int64_t j = 0; j < out; j++) {
                    const float * row = Wh.data() + (size_t) j * (size_t) in;
                    double        acc = 0.0;
                    for (int64_t i = 0; i < in; i++) {
                        acc += (double) row[(size_t) i] * xr[(size_t) i];
                    }
                    ref[(size_t) j] = acc;
                }
            }

            // Scale by the larger of the two, not by the reference alone: at
            // step 0 both parameterizations have a delta of EXACTLY zero by
            // construction, and dividing by that would turn agreement into an
            // infinite relative error.
            double mx = 0.0, mg = 0.0;
            for (int64_t j = 0; j < out; j++) {
                mx = std::max(mx, std::fabs(got[(size_t) j] - ref[(size_t) j]));
                mg = std::max(mg, std::max(std::fabs(ref[(size_t) j]), std::fabs(got[(size_t) j])));
            }
            ref_mag    = std::max(ref_mag, mg);
            site_worst = std::max(site_worst, mx / std::max(1e-12, mg));
        }
        worst = std::max(worst, site_worst);
        checked++;
        fprintf(stderr, "[verify-delta] layer %d slot %-16s in %5lld out %5lld rank %3lld scale %.4f: worst rel %.3e (|ref| %.3e)\n",
                layer, lm_slot_peft_name(s), (long long) in, (long long) out, (long long) rr,
                (double) p.base_scale, site_worst, ref_mag);
        if (site_worst > BAR) {
            bad++;
        }
    }

    fprintf(stderr, "[verify-delta] %s: %s, %d site(s), worst rel %.3e (bar %.1e), %d failure(s)\n",
            bad ? "FAIL" : "PASS", L.pissa ? "PiSSA rank-2r export vs s(BA - B0A0)" : "HRA export vs W(R - I)",
            checked, worst, BAR, bad);
    mm3_lm_adapter_free(ad);
    if (bad) {
        *err = "the exported adapter does not reproduce the training graph's delta — see the lines above";
        return false;
    }
    return checked > 0;
}

/** Load `dir`'s adapter_model.safetensors with the runtime loader and compare
 *  it against the live trainer bank. Returns true on PASS; prints one line per
 *  failed check plus a verdict line. */
static bool mm3_lm_verify_export(const std::string & dir, const LmLora & L, std::string * err,
                                 const std::string & base_lm_path = "") {
    const std::string sf = dir + "/adapter_model.safetensors";
    std::string       lerr;
    MM3LmAdapter *    ad = mm3_lm_adapter_load(sf.c_str(), &lerr, base_lm_path.empty() ? nullptr : base_lm_path.c_str());
    if (!ad) {
        *err = "the runtime loader refused the file we just wrote: " + lerr;
        return false;
    }

    int    checked = 0, bad = 0;
    double worst_ab = 0.0, worst_m = 0.0, worst_scale = 0.0;

    // f16 has an 11-bit significand: a relative error of ~2^-11 is the floor,
    // and anything materially above it means the bytes changed meaning, not
    // precision.
    const double F16_REL = 1.5e-3;

    std::vector<float> tr, rt;
    auto pull_f32 = [&](ggml_tensor * t, std::vector<float> * v) {
        v->assign((size_t) ggml_nelements(t), 0.0f);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, v->data(), 0, v->size() * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(h.data(), v->data(), (int64_t) h.size());
        } else {
            v->clear();
        }
    };

    for (int l = L.layer_lo; l < L.layer_hi && l < MM3_LM_ADAPTER_LAYERS; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            const QwLoraPair &       q = L.layers[l].p[s];
            const MM3LmAdapterPair & p = ad->mods[l][s];
            if (!q.A || !q.B) {
                continue;
            }
            if (!p.has_lora()) {
                fprintf(stderr, "[verify-export] layer %d slot %s: the loader has no pair for a trained site\n", l,
                        lm_slot_peft_name(s));
                bad++;
                continue;
            }
            // 1. scale. This is the check rsLoRA exists to fail.
            const double ds = std::fabs((double) p.base_scale - (double) q.scale) /
                              std::max(1e-9, (double) std::fabs(q.scale));
            worst_scale = std::max(worst_scale, ds);
            if (ds > 1e-4) {
                fprintf(stderr,
                        "[verify-export] layer %d slot %s: loader scale %.6f but the graph applied %.6f\n", l,
                        lm_slot_peft_name(s), (double) p.base_scale, (double) q.scale);
                bad++;
            }
            // 2. tensors, within f16 rounding.
            ggml_tensor * pair_tr[2] = { q.A, q.B };
            ggml_tensor * pair_rt[2] = { p.a, p.b };
            for (int k = 0; k < 2; k++) {
                pull_f32(pair_tr[k], &tr);
                pull_f32(pair_rt[k], &rt);
                if (tr.size() != rt.size() || tr.empty()) {
                    fprintf(stderr, "[verify-export] layer %d slot %s: element count %zu vs %zu\n", l,
                            lm_slot_peft_name(s), tr.size(), rt.size());
                    bad++;
                    continue;
                }
                double mx = 0.0, scale = 0.0;
                for (size_t i = 0; i < tr.size(); i++) {
                    mx    = std::max(mx, (double) std::fabs(tr[i] - rt[i]));
                    scale = std::max(scale, (double) std::fabs(tr[i]));
                }
                const double rel = mx / std::max(1e-9, scale);
                worst_ab         = std::max(worst_ab, rel);
                if (rel > F16_REL) {
                    fprintf(stderr, "[verify-export] layer %d slot %s %s: max rel %.3e over the f16 floor\n", l,
                            lm_slot_peft_name(s), k ? "B" : "A", rel);
                    bad++;
                }
            }
            // 3. DoRA magnitudes. F32 on both sides, so this one is exact.
            if (L.dora) {
                if (!p.m) {
                    fprintf(stderr, "[verify-export] layer %d slot %s: DoRA magnitude missing after load\n", l,
                            lm_slot_peft_name(s));
                    bad++;
                } else {
                    pull_f32(q.m, &tr);
                    pull_f32(p.m, &rt);
                    double mx = 0.0, scale = 0.0;
                    for (size_t i = 0; i < tr.size() && i < rt.size(); i++) {
                        mx    = std::max(mx, (double) std::fabs(tr[i] - rt[i]));
                        scale = std::max(scale, (double) std::fabs(tr[i]));
                    }
                    const double rel = mx / std::max(1e-9, scale);
                    worst_m          = std::max(worst_m, rel);
                    if (rel > 1e-6) {
                        fprintf(stderr, "[verify-export] layer %d slot %s: DoRA magnitude rel %.3e\n", l,
                                lm_slot_peft_name(s), rel);
                        bad++;
                    }
                }
            }
            checked++;
        }
    }

    if (L.dora && ad->dora_n != checked) {
        fprintf(stderr, "[verify-export] the loader found %d DoRA magnitudes for %d trained sites\n", ad->dora_n,
                checked);
        bad++;
    }
    if (L.loha && !ad->is_loha) {
        fprintf(stderr, "[verify-export] a LoHa export did not load as a LoHa\n");
        bad++;
    }
    if (L.hira && !ad->is_hira) {
        fprintf(stderr, "[verify-export] a HiRA export did not load as a HiRA (peft_type)\n");
        bad++;
    }

    fprintf(stderr,
            "[verify-export] %s: %d sites, scale worst rel %.3e (bar 1e-4), A/B worst rel %.3e (f16 bar %.1e), "
            "DoRA magnitude worst rel %.3e, %d failure(s)\n",
            bad ? "FAIL" : "PASS", checked, worst_scale, worst_ab, F16_REL, worst_m, bad);
    mm3_lm_adapter_free(ad);
    if (bad) {
        *err = "export/load round trip failed — see the lines above";
        return false;
    }
    return checked > 0;
}
