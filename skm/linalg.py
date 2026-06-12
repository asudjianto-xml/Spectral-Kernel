"""Krylov linear algebra for the kernel-ridge decode.

The fitted MS-SKM predicts through full-train kernel ridge regression,
``alpha = (K + lambda I)^{-1} Y``. For n in the tens of thousands a dense
factorization is wasteful: the spectral kernel is effectively low rank, so a
truncated Lanczos tridiagonalization recovers the ridge solution from a handful
of matrix-vector products. The solve is d_eff-immune -- the rank is a compute
budget, not a model hyperparameter -- and one Lanczos basis is reused across a
sweep of ridge values.
"""
from __future__ import annotations

import torch


def lanczos(matvec, b, k, device, dtype):
    """Lanczos tridiagonalization of a symmetric operator given by ``matvec``.

    Builds an orthonormal Krylov basis Q (n, m) and the tridiagonal coefficients
    (alpha, beta) such that ``Q^T A Q = tridiag(beta, alpha, beta)``, starting
    from b. Full reorthogonalization (twice) keeps Q orthonormal in float
    arithmetic. Returns ``(Q, alpha, beta, ||b||)``; m <= k may be smaller if the
    Krylov space deflates.
    """
    n = b.shape[0]
    Q = torch.zeros(n, k, dtype=dtype, device=device)
    alpha = torch.zeros(k, dtype=dtype, device=device)
    beta = torch.zeros(max(k - 1, 1), dtype=dtype, device=device)
    nb = torch.linalg.norm(b)
    Q[:, 0] = b / nb
    last = k
    for j in range(k):
        w = matvec(Q[:, j])
        alpha[j] = Q[:, j] @ w
        w = w - alpha[j] * Q[:, j] - (beta[j - 1] * Q[:, j - 1] if j > 0 else 0.0)
        w = w - Q[:, :j + 1] @ (Q[:, :j + 1].t() @ w)      # reorthogonalize twice
        w = w - Q[:, :j + 1] @ (Q[:, :j + 1].t() @ w)
        if j < k - 1:
            beta[j] = torch.linalg.norm(w)
            if beta[j] < 1e-10:                            # Krylov space exhausted
                last = j + 1
                break
            Q[:, j + 1] = w / beta[j]
    return Q[:, :last], alpha[:last], beta[:last - 1], nb


def krr_solve(K, Y, lam, rank, device, dtype):
    """Kernel-ridge coefficients ``alpha = (K + lam I)^{-1} Y`` via Lanczos.

    K is the (n, n) train Gram, Y is (n, C). Each output column is solved on its
    own Lanczos basis. Returns alpha (n, C) so a prediction is ``K(x, train) @ alpha``.
    """
    cols = []
    for c in range(Y.shape[1]):
        Q, al, be, nb = lanczos(lambda v: K @ v, Y[:, c].contiguous(), rank, device, dtype)
        Tk = torch.diag(al) + torch.diag(be, 1) + torch.diag(be, -1)
        theta, U = torch.linalg.eigh(Tk)
        cols.append(Q @ (nb * (U @ (U[0, :] / (theta + lam)))))
    return torch.stack(cols, 1)


def krr_solve_sweep(Ktt, Kqt, Y, lambdas, device, dtype):
    """KRR with a ridge sweep, reusing one Lanczos basis per output column.

    Returns ``(preds, train_coef)`` for every lambda in ``lambdas``: ``preds[i]``
    is the (q, C) prediction K(query, train) @ alpha_i, and ``train_coef[i]`` is the
    (n, C) coefficient. Used to pick the ridge on a validation fold without
    re-factorizing the kernel.
    """
    rank = min(Ktt.shape[0], 150)
    per_col = []
    for c in range(Y.shape[1]):
        Q, al, be, nb = lanczos(lambda v: Ktt @ v, Y[:, c].contiguous(), rank, device, dtype)
        Tk = torch.diag(al) + torch.diag(be, 1) + torch.diag(be, -1)
        theta, U = torch.linalg.eigh(Tk)
        per_col.append((Q, theta, U, U[0, :], nb, Kqt @ Q))
    preds, coefs = [], []
    for lam in lambdas:
        coef = torch.stack([Q @ (nb * (U @ (u0 / (theta + lam))))
                            for (Q, theta, U, u0, nb, _) in per_col], 1)
        pred = torch.stack([Mq @ (nb * (U @ (u0 / (theta + lam))))
                            for (Q, theta, U, u0, nb, Mq) in per_col], 1)
        coefs.append(coef)
        preds.append(pred)
    return preds, coefs
