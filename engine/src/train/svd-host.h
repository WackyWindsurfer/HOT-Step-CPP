#pragma once
// svd-host.h — the two host-side linear-algebra kernels the randomized SVD
// needs, shared by every trainer that does a PiSSA init or an HRA export.
//
// They lived in dit-pissa.h until 2026-09-05, when the LM trainers needed the
// same maths and could not include a DiT header to get it (dit-pissa.h pulls in
// dit-adapter.h, which pulls in the whole DiT model). The bodies are unchanged
// from that file; dit-pissa.h now forwards to them, so there is one QR and one
// eigensolver in the tree rather than a DiT copy and an LM copy that drift.

#include <cmath>
#include <cstddef>
#include <vector>

// Modified Gram-Schmidt on a column-major out x q matrix (column j at j*out).
// Two passes: the second removes the loss of orthogonality the first leaves
// when the power iteration has made the columns nearly parallel.
static void hs_qr_mgs(std::vector<double> & Y, int out, int q) {
    for (int pass = 0; pass < 2; pass++) {
        for (int j = 0; j < q; j++) {
            double * cj = Y.data() + (size_t) j * (size_t) out;
            for (int i = 0; i < j; i++) {
                const double * ci  = Y.data() + (size_t) i * (size_t) out;
                double         dot = 0.0;
                for (int k = 0; k < out; k++) {
                    dot += ci[k] * cj[k];
                }
                for (int k = 0; k < out; k++) {
                    cj[k] -= dot * ci[k];
                }
            }
            double nrm = 0.0;
            for (int k = 0; k < out; k++) {
                nrm += cj[k] * cj[k];
            }
            nrm = sqrt(nrm);
            if (nrm < 1e-300) {
                // Degenerate column: replace with a unit vector on an axis that
                // no earlier column occupies (only reachable for rank-deficient W).
                for (int k = 0; k < out; k++) {
                    cj[k] = 0.0;
                }
                cj[j % out] = 1.0;
                continue;
            }
            for (int k = 0; k < out; k++) {
                cj[k] /= nrm;
            }
        }
    }
}

// Cyclic Jacobi on a symmetric q x q matrix G (row-major, overwritten).
// evecs is column-major (eigenvector k at k*q); evals[k] pairs with it.
static void hs_jacobi_sym(std::vector<double> & G, int q, std::vector<double> & evals,
                          std::vector<double> & evecs) {
    evecs.assign((size_t) q * (size_t) q, 0.0);
    for (int i = 0; i < q; i++) {
        evecs[(size_t) i * (size_t) q + (size_t) i] = 1.0;
    }
    auto at = [&](int r, int c) -> double & { return G[(size_t) r * (size_t) q + (size_t) c]; };
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (int r = 0; r < q; r++) {
            for (int c = r + 1; c < q; c++) {
                off += at(r, c) * at(r, c);
            }
        }
        if (off < 1e-22) {
            break;
        }
        for (int p = 0; p < q; p++) {
            for (int r = p + 1; r < q; r++) {
                const double apq = at(p, r);
                if (fabs(apq) < 1e-300) {
                    continue;
                }
                const double theta = (at(r, r) - at(p, p)) / (2.0 * apq);
                const double t     = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                const double cs    = 1.0 / sqrt(t * t + 1.0);
                const double sn    = t * cs;
                for (int k = 0; k < q; k++) {
                    const double gkp = at(k, p), gkr = at(k, r);
                    at(k, p) = cs * gkp - sn * gkr;
                    at(k, r) = sn * gkp + cs * gkr;
                }
                for (int k = 0; k < q; k++) {
                    const double gpk = at(p, k), grk = at(r, k);
                    at(p, k) = cs * gpk - sn * grk;
                    at(r, k) = sn * gpk + cs * grk;
                }
                for (int k = 0; k < q; k++) {
                    double & vkp = evecs[(size_t) p * (size_t) q + (size_t) k];
                    double & vkr = evecs[(size_t) r * (size_t) q + (size_t) k];
                    const double a = vkp, b = vkr;
                    vkp = cs * a - sn * b;
                    vkr = sn * a + cs * b;
                }
            }
        }
    }
    evals.resize((size_t) q);
    for (int i = 0; i < q; i++) {
        evals[(size_t) i] = at(i, i);
    }
}

// Apply r Householder reflections, in graph order, to a column-major set of `n`
// vectors of length `in` (vector j at j*in). V is column-major [in, r]:
// x <- H_{r-1} ... H_0 x with H_k = I - 2 v_k v_k^T / ||v_k||^2.
static void hs_householder_apply(const std::vector<float> & V, int in, int r, std::vector<double> & X, int n) {
    for (int k = 0; k < r; k++) {
        const float * v  = V.data() + (size_t) k * (size_t) in;
        double        n2 = 0.0;
        for (int i = 0; i < in; i++) {
            n2 += (double) v[i] * (double) v[i];
        }
        if (n2 <= 0.0) {
            continue;
        }
        for (int j = 0; j < n; j++) {
            double * x   = X.data() + (size_t) j * (size_t) in;
            double   dot = 0.0;
            for (int i = 0; i < in; i++) {
                dot += (double) v[i] * x[i];
            }
            const double f = 2.0 * dot / n2;
            for (int i = 0; i < in; i++) {
                x[i] -= f * (double) v[i];
            }
        }
    }
}
