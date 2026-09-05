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

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

/** Load `dir`'s adapter_model.safetensors with the runtime loader and compare
 *  it against the live trainer bank. Returns true on PASS; prints one line per
 *  failed check plus a verdict line. */
static bool mm3_lm_verify_export(const std::string & dir, const LmLora & L, std::string * err) {
    const std::string sf = dir + "/adapter_model.safetensors";
    std::string       lerr;
    MM3LmAdapter *    ad = mm3_lm_adapter_load(sf.c_str(), &lerr);
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
