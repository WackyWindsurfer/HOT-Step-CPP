#pragma once
// lm-resume.h — continue LM adapter training from an exported run
// (`ace-train train-lm --init-adapter <dir>`), 2026-08-09.
//
// Why: training stops on a uniform target loss, but the eval tool
// (server/scripts/lm-adapter-eval.ts) showed the *distribution shift toward
// the artist* peaks at a per-artist point that loss cannot see. The strategy
// is train-past-then-select: resume an existing adapter with a lower target
// and milestone snapshots on, then pick the best snapshot by eval. Resume
// pays only for the NEW epochs — the existing 183-adapter corpus keeps its
// runs as starting points instead of retraining from scratch.
//
// What resuming means here: the exported factors are the complete trainable
// state; optimizer momentum is not saved, so a resumed run is a fine-tune
// continuation (fresh warmup + cosine), not a bit-exact "never stopped" run.
// Exports are BF16 for LoKr (F32 for PEFT LoRA), so LoKr resume also eats one
// BF16 rounding of the factors. Both are deliberate v1 trade-offs.
//
// REFUSE RULES (lm-export.h §2.3 "RESUME RULE" made real). Identity
// hyperparameters come FROM THE SOURCE RUN's lm_train_log.json; a CLI value
// that contradicts the source is a hard exit 2, never a coercion:
//   - adapter_type, and its shape params (rank/alpha | lokr dim/alpha/factor):
//     different values would allocate tensors the file cannot fill.
//   - weights (f32-window|bf16): under bf16 the gradients are not the same
//     quantity (S6) — continuing across modes silently changes what "loss
//     4.0 → 1.5" means.
//   - lm_size: a 4B adapter on a 0.6B base dies late and confusingly
//     ("36 layers but model has 28"); refuse up front. A different file of
//     the SAME size only warns — quant/bf16 variants of one base are fine.
//
// NOT identity, deliberately: `bwd` and `attn` (--attn exact|flash|flash-f32,
// D7). Both change the SUMMATION ORDER of a gradient, not the gradient — flash
// recomputes the softmax from Q/K/LSE in tiles instead of reading back a
// retained array, and over 200 same-seed DiT epochs it drifted LESS than
// `--bwd mm` does. So neither is read from the source log, neither is adopted,
// and a CLI value that differs from the source run's is not a contradiction to
// refuse. They are RECORDED in lm_train_log.json (lm-export.h attn_mode /
// attn_prec) so a finished run can still say which it used; `weights` is the
// only lever in this family that IS a barrier, because bf16 rounds the
// activation gradient at every layer and changes what the loss curve means.
//
// The honest gate: a resumed run's FIRST epoch loss should land near the
// source's saved_loss (teacher-forced, same data). lm-train-run.h logs the
// comparison; a large gap means the init mapping or the dataset changed.

#include "safetensors.h"
#include "train/lm-graph.h"
#include "yyjson.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// ─── source-run identity (from lm_train_log.json) ───────────────────────────

struct LmResumeSource {
    std::string dir;                    // the --init-adapter dir
    std::string adapter_type = "lora";
    // Post-LoRA parameterization: "lora" | "dora" | "hira" | "loha" | "hra" |
    // "pissa" (the last of which lm_resume_prepare refuses outright)
    // (lm-export.h `param_method`). Identity, not a knob — a DoRA adapter
    // resumed as a plain LoRA would silently drop its magnitudes.
    std::string method       = "lora";
    // alpha/sqrt(r) instead of alpha/r. Identity too, in the way that matters
    // for a chain: it changes nothing about the tensor set, so a leg that drops
    // it loads the same A/B and applies them 16x weaker at r256, then exports
    // use_rslora:false so the runtime does the same. Nothing else would notice.
    bool        rslora       = false;
    int         rank = 0, alpha = 0;
    int         lokr_dim = 0, lokr_factor = 0;
    float       lokr_alpha = 0.0f;
    bool        lokr_decompose = true;
    std::string weights;                // f32-window|bf16
    std::string lm_size, lm_path;
    std::string trigger, trigger_position;
    double      saved_loss  = -1.0;
    int         saved_epoch = 0;
    // Prodigy's final step-size estimate from the source run (0 = not prodigy
    // or an older log). Adopted as this leg's --prodigy-d0 unless the user set
    // one, so a chained leg does not restart its warm-up from 1e-6.
    double      prodigy_d   = 0.0;
};

/** Which identity flags the user typed explicitly (tracked by cmd_train_lm's
 *  parser). Explicit-and-different from the source → refuse; omitted → adopt. */
struct LmResumeExplicit {
    bool rank = false, alpha = false, adapter_type = false;
    bool method = false;  // --dora / --hira / --loha typed on this run's CLI
    bool rslora = false;  // --rslora typed on this run's CLI
    bool lokr_dim = false, lokr_alpha = false, lokr_factor = false;
    bool weights = false;
    bool prodigy_d0 = false;   // not identity: only decides whether the source's d is adopted
};

static bool lm_resume_read_log(const std::string & dir, LmResumeSource * src, std::string * err) {
    const std::string log_path = dir + "/lm_train_log.json";
    yyjson_doc * doc = yyjson_read_file(log_path.c_str(), 0, NULL, NULL);
    if (!doc) {
        *err = "cannot read " + log_path + " — resume needs the source run's recorded config";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * cfg  = yyjson_obj_get(root, "config");
    if (!cfg || !yyjson_is_obj(cfg)) {
        yyjson_doc_free(doc);
        *err = log_path + " has no config object";
        return false;
    }
    auto s = [&](const char * k) -> std::string {
        yyjson_val * v = yyjson_obj_get(cfg, k);
        return yyjson_is_str(v) ? yyjson_get_str(v) : "";
    };
    auto i = [&](const char * k, int dflt) -> int {
        yyjson_val * v = yyjson_obj_get(cfg, k);
        return yyjson_is_num(v) ? (int) yyjson_get_num(v) : dflt;
    };
    src->dir              = dir;
    src->adapter_type     = s("adapter_type").empty() ? "lora" : s("adapter_type");
    src->method           = s("param_method").empty() ? "lora" : s("param_method");
    {
        yyjson_val * v = yyjson_obj_get(cfg, "rslora");
        src->rslora    = yyjson_is_bool(v) ? yyjson_get_bool(v) : false;
    }
    src->rank             = i("rank", 0);
    src->alpha            = i("alpha", 0);
    src->lokr_dim         = i("lokr_dim", 0);
    src->lokr_factor      = i("lokr_factor", 0);
    {
        yyjson_val * v  = yyjson_obj_get(cfg, "lokr_alpha");
        src->lokr_alpha = yyjson_is_num(v) ? (float) yyjson_get_num(v) : 0.0f;
    }
    src->weights          = s("weights").empty() ? "f32-window" : s("weights");
    {
        yyjson_val * v = yyjson_obj_get(cfg, "prodigy_d");
        src->prodigy_d = yyjson_is_num(v) ? yyjson_get_num(v) : 0.0;
    }
    src->lm_size          = s("lm_size");
    src->lm_path          = s("lm_path");
    src->trigger          = s("trigger");
    src->trigger_position = s("trigger_position");
    {
        yyjson_val * v   = yyjson_obj_get(root, "saved_loss");
        src->saved_loss  = yyjson_is_num(v) ? yyjson_get_num(v) : -1.0;
        yyjson_val * e   = yyjson_obj_get(root, "saved_epoch");
        src->saved_epoch = yyjson_is_num(e) ? (int) yyjson_get_num(e) : 0;
    }
    yyjson_doc_free(doc);
    return true;
}

// ─── weight loading (the exporters inverted) ────────────────────────────────

/** BF16 → F32 widening (bf16 is the top half of the f32 bit pattern). */
static void lm_resume_bf16_to_f32(const uint16_t * in, float * out, size_t n) {
    for (size_t j = 0; j < n; j++) {
        const uint32_t u = ((uint32_t) in[j]) << 16;
        memcpy(&out[j], &u, sizeof(float));
    }
}

/** Fetch one tensor from `st` into `dst` (an F32 trainer param). The exporters
 *  wrote raw ggml memory with the shape recorded as {ne1, ne0}, so byte order
 *  matches exactly — only the dtype may need widening. */
static bool lm_resume_fill(const STFile & st, const char * name, ggml_tensor * dst, std::string * err) {
    const STEntry * e = st_find(st, name);
    if (!e) {
        *err = std::string("tensor missing from adapter file: ") + name;
        return false;
    }
    int64_t n = 1;
    for (int d = 0; d < e->n_dims; d++) {
        n *= e->shape[d];
    }
    if (n != ggml_nelements(dst)) {
        char b[224];
        snprintf(b, sizeof(b), "%s holds %lld values but the trainer tensor wants %lld — source run shape mismatch",
                 name, (long long) n, (long long) ggml_nelements(dst));
        *err = b;
        return false;
    }
    const void * data = st_data(st, *e);
    if (e->dtype == "F32") {
        ggml_backend_tensor_set(dst, data, 0, (size_t) n * sizeof(float));
    } else if (e->dtype == "BF16") {
        std::vector<float> f((size_t) n);
        lm_resume_bf16_to_f32((const uint16_t *) data, f.data(), (size_t) n);
        ggml_backend_tensor_set(dst, f.data(), 0, f.size() * sizeof(float));
    } else {
        *err = std::string(name) + " has unsupported dtype " + e->dtype;
        return false;
    }
    return true;
}

/** Load an exported adapter's factors into the freshly-initialized trainer
 *  params. Every expected tensor must be present — a partial adapter would
 *  train from a silently-wrong start. Returns the number of tensors loaded. */
static bool lm_resume_load(LmLora * L, const std::string & dir, int * n_loaded, std::string * err) {
    if (L->hra) {
        // The exported LoRA is (W U) Q^T — an orthonormal BASIS of the vectors'
        // span, not the vectors, so it cannot be inverted back into them. The
        // sidecar lm_hra_write_vectors left beside it is the resume format.
        const std::string vpath = dir + "/hra_vectors.safetensors";
        STFile            vst;
        if (!st_open(&vst, vpath.c_str())) {
            *err = "cannot open " + vpath + " — an HRA run resumes from its vector sidecar, not the adapter";
            return false;
        }
        int  vcount = 0;
        bool vok    = true;
        for (int l = L->layer_lo; l < L->layer_hi && vok; l++) {
            for (int s = 0; s < QW_LORA_NSLOTS && vok; s++) {
                QwLoraPair & pr = L->layers[l].p[s];
                if (!pr.A) {
                    continue;
                }
                char nm[96];
                snprintf(nm, sizeof(nm), "L%d.s%d.hra_v", l, s);
                vok = lm_resume_fill(vst, nm, pr.A, err);
                vcount++;
            }
        }
        st_close(&vst);
        if (vok && n_loaded) {
            *n_loaded = vcount;
        }
        return vok;
    }
    const std::string path =
        dir + (L->is_lokr ? "/lokr_weights.safetensors" : "/adapter_model.safetensors");
    STFile st;
    if (!st_open(&st, path.c_str())) {
        *err = "cannot open " + path;
        return false;
    }
    int  count = 0;
    bool ok    = true;
    for (int l = L->layer_lo; l < L->layer_hi && ok; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS && ok; s++) {
            QwLoraPair & pr = L->layers[l].p[s];
            if (L->is_lokr) {
                if (!pr.has_lokr()) {
                    continue;
                }
                std::string site(lm_slot_peft_name(s));
                for (size_t j = 0; j < site.size(); j++) {
                    if (site[j] == '.') {
                        site[j] = '_';
                    }
                }
                char stem[128];
                snprintf(stem, sizeof(stem), "lycoris_layers_%d_%s", l, site.c_str());

                // The stored alpha must reproduce the scale this run will
                // apply — a mismatch means the file came from a different
                // dim/alpha config than the log claimed.
                const STEntry * ae = st_find(st, (std::string(stem) + ".alpha").c_str());
                if (ae) {
                    float av = 0.0f;
                    if (ae->dtype == "F32") {
                        memcpy(&av, st_data(st, *ae), sizeof(float));
                    } else if (ae->dtype == "BF16") {
                        uint16_t u16;
                        memcpy(&u16, st_data(st, *ae), sizeof(uint16_t));
                        lm_resume_bf16_to_f32(&u16, &av, 1);
                    }
                    const float want = pr.lokr_scale * (float) L->lokr_dim;
                    if (fabsf(av - want) > 0.01f * fmaxf(1.0f, fabsf(want))) {
                        char b[192];
                        snprintf(b, sizeof(b), "%s.alpha is %.4g but this config implies %.4g — wrong source file?",
                                 stem, (double) av, (double) want);
                        *err = b;
                        ok   = false;
                        break;
                    }
                }
                ok = ok && lm_resume_fill(st, (std::string(stem) + ".lokr_w1").c_str(), pr.w1, err);
                count++;
                if (ok && pr.w2) {
                    ok = lm_resume_fill(st, (std::string(stem) + ".lokr_w2").c_str(), pr.w2, err);
                    count++;
                } else if (ok) {
                    ok = lm_resume_fill(st, (std::string(stem) + ".lokr_w2_a").c_str(), pr.w2_a, err) &&
                         lm_resume_fill(st, (std::string(stem) + ".lokr_w2_b").c_str(), pr.w2_b, err);
                    count += 2;
                }
            } else if (pr.has_loha()) {
                // LyCORIS layout, the inverse of lm_export_peft's LoHa branch:
                // hada_w1_a = B, hada_w1_b = A, hada_w2_a = B2, hada_w2_b = A2.
                const std::string stem      = lm_lycoris_key_stem(l, s);
                ggml_tensor *     four[4]   = { pr.B, pr.A, pr.B2, pr.A2 };
                const char *      sfx[4]    = { "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b" };
                for (int k = 0; k < 4 && ok; k++) {
                    ok = lm_resume_fill(st, (stem + "." + sfx[k]).c_str(), four[k], err);
                    count++;
                }
            } else {
                if (!pr.A || !pr.B) {
                    continue;
                }
                char nm[192];
                snprintf(nm, sizeof(nm), "base_model.model.model.layers.%d.%s.lora_A.weight", l,
                         lm_slot_peft_name(s));
                ok = lm_resume_fill(st, nm, pr.A, err);
                if (ok) {
                    snprintf(nm, sizeof(nm), "base_model.model.model.layers.%d.%s.lora_B.weight", l,
                             lm_slot_peft_name(s));
                    ok = lm_resume_fill(st, nm, pr.B, err);
                }
                count += 2;
                // DoRA magnitudes. Every expected tensor must be present — a
                // resumed DoRA that quietly fell back to ||W||_col would be a
                // different adapter than the one that was saved. `nrm` is NOT
                // read: it is re-derived from the resumed A/B at the first
                // optimizer window (lm_lora_dora_refresh).
                if (ok && pr.m) {
                    snprintf(nm, sizeof(nm),
                             "base_model.model.model.layers.%d.%s.lora_magnitude_vector.weight", l,
                             lm_slot_peft_name(s));
                    ok = lm_resume_fill(st, nm, pr.m, err);
                    count++;
                }
            }
        }
    }
    st_close(&st);
    if (ok && n_loaded) {
        *n_loaded = count;
    }
    return ok;
}

// ─── CLI-side prepare: adopt-or-refuse ──────────────────────────────────────

/** Called by cmd_train_lm after flag parsing. Reads the source run's config,
 *  refuses explicit contradictions, adopts everything else. `errbuf` gets a
 *  ready-to-print message on false. */
template <typename ArgsT>
static bool lm_resume_prepare(ArgsT * a, const LmResumeExplicit & saw, LmResumeSource * src, std::string * errbuf) {
    if (!lm_resume_read_log(a->init_adapter, src, errbuf)) {
        return false;
    }

    struct Conflict {
        const char * flag;
        std::string  cli, source;
    };
    std::vector<Conflict> bad;
    auto check_i = [&](bool explicit_set, const char * flag, int cli, int source) {
        if (explicit_set && source > 0 && cli != source) {
            bad.push_back({ flag, std::to_string(cli), std::to_string(source) });
        }
    };
    if (saw.adapter_type && a->adapter_type != src->adapter_type) {
        bad.push_back({ "--adapter-type", a->adapter_type, src->adapter_type });
    }
    if (saw.weights && a->weights != src->weights) {
        bad.push_back({ "--weights", a->weights, src->weights });
    }
    // Parameterization identity. Typed-and-different is a refusal; omitted is
    // adopted below, so `--init-adapter <a DoRA run>` continues as DoRA without
    // the user having to remember to re-type --dora.
    // PiSSA's whole point is a starting basis derived from the base weight plus
    // a frozen negative copy of it. The exported rank-2r file has already
    // folded the two halves together, so there is nothing to recover A0/B0
    // from — and re-running the init on a run in progress would move the base
    // the trained factors were fitted against. Refuse rather than resume into a
    // silently different adapter.
    if (src->method == "pissa") {
        *errbuf = "--init-adapter " + a->init_adapter +
                  " was trained with --pissa, which cannot be resumed: the rank-2r export has already folded the "
                  "init factors into the delta, and re-deriving them would move the residual the trained factors "
                  "were fitted against. Start a fresh run.";
        return false;
    }
    const std::string cli_method = a->hra    ? "hra"
                                   : a->hira ? "hira"
                                   : a->loha ? "loha"
                                   : a->dora ? "dora"
                                             : "lora";
    if (saw.method && cli_method != src->method) {
        bad.push_back({ "--dora/--hira/--loha/--hra", cli_method, src->method });
    }
    if (saw.rslora && a->rslora != src->rslora) {
        bad.push_back({ "--rslora", a->rslora ? "on" : "off", src->rslora ? "on" : "off" });
    }
    if (a->pissa) {
        *errbuf = "--pissa cannot be combined with --init-adapter: the init rewrites A/B from the base weight's "
                  "singular vectors, which would discard everything the source run learned.";
        return false;
    }
    check_i(saw.rank, "--rank", a->rank, src->rank);
    check_i(saw.alpha, "--alpha", a->alpha, src->alpha);
    check_i(saw.lokr_dim, "--lokr-dim", a->lokr_dim, src->lokr_dim);
    check_i(saw.lokr_factor, "--lokr-factor", a->lokr_factor, src->lokr_factor);
    if (saw.lokr_alpha && src->lokr_alpha > 0.0f && fabsf(a->lokr_alpha - src->lokr_alpha) > 1e-3f) {
        bad.push_back({ "--lokr-alpha", std::to_string(a->lokr_alpha), std::to_string(src->lokr_alpha) });
    }
    if (!bad.empty()) {
        std::string m = "--init-adapter refuses to continue " + a->init_adapter +
                        " with a different identity than it was trained with:\n";
        for (size_t j = 0; j < bad.size(); j++) {
            m += "  " + std::string(bad[j].flag) + " " + bad[j].cli + " but the source run used " + bad[j].source +
                 "\n";
        }
        m += "Drop the conflicting flag(s) — resume adopts the source run's identity.";
        *errbuf = m;
        return false;
    }

    // Adopt. Shape identity always comes from the source; sampling knobs
    // (lr, epochs, target, schedule) stay whatever this run asked for.
    a->adapter_type = src->adapter_type;
    a->dora = (src->method == "dora");
    a->hira = (src->method == "hira");
    a->loha = (src->method == "loha");
    a->hra  = (src->method == "hra");
    // Adopted for the same reason as the method: the server's multi-stage
    // spawner types --rslora only on the leg that turned it on, so every later
    // --init-adapter leg would otherwise continue at alpha/r.
    a->rslora = src->rslora;
    if (src->rank > 0) {
        a->rank = src->rank;
    }
    if (src->alpha > 0) {
        a->alpha = src->alpha;
    }
    if (src->lokr_dim > 0) {
        a->lokr_dim = src->lokr_dim;
    }
    if (src->lokr_factor > 0) {
        a->lokr_factor = src->lokr_factor;
    }
    if (src->lokr_alpha > 0.0f) {
        a->lokr_alpha = src->lokr_alpha;
    }
    if (!src->weights.empty()) {
        a->weights = src->weights;
    }
    // Prodigy: carry the learned step size into this leg. Not an identity
    // check — a run that switched optimizer simply ignores it.
    if (a->optimizer == "prodigy" && !saw.prodigy_d0 && src->prodigy_d > 0.0) {
        a->prodigy_d0 = src->prodigy_d;
        fprintf(stderr, "[train-lm] resume: adopting the source run's Prodigy step size d = %.3g as --prodigy-d0\n",
                src->prodigy_d);
    }
    if (a->trigger.empty() && !src->trigger.empty()) {
        a->trigger          = src->trigger;
        a->trigger_position = src->trigger_position;
    }
    return true;
}
