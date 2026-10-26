"""Projected-gradient solver for the source-aware ellipsoidal boundary.

For each known class ``c`` the acceptance score is

    s_c(z) = b_c - ||x_c||^2 - sum_m (theta_cm - theta0_m) * Mf_cm(z),

with ``x_c = (z - mu_c) / delta_c``. The shape matrix is

    Q_c = U_c H_c U_c^T + t_c (I - U_c U_c^T),

where ``U_c`` contains the top-r principal directions of the standardized seed
representations and ``H_c`` is a symmetric positive definite matrix whose
spectrum is constrained to ``[1/kappa, kappa]``. The tail scale ``t_c`` lives
in ``[tail_min, tail_max]`` and ``b_c`` in ``[b_min, b_max]``.
"""
import numpy as np

from .inputs import _sigmoid, _sp, metric_features, seed_stats


def canonical_block_basis(block, tol=1e-8):
    d, m = block.shape
    if m == 0:
        return np.zeros((d, 0), dtype=np.float64)
    accepted = []
    for j in range(d):
        v = block @ block[j, :]
        for _ in range(2):
            for u in accepted:
                v = v - float(u @ v) * u
        norm = float(np.linalg.norm(v))
        if norm <= tol:
            continue
        v = v / norm
        pivot = int(np.argmax(np.abs(v)))
        if v[pivot] < 0:
            v = -v
        accepted.append(v)
        if len(accepted) == m:
            break
    if len(accepted) != m:
        raise ValueError('could not construct a canonical basis for a degenerate block')
    return np.stack(accepted, axis=1)


def canonical_subspace_perclass(z_seed, y_seed, mu, delta, r=16, tol_rel=1e-8):
    K = int(y_seed.max()) + 1
    d = z_seed.shape[1]
    subspaces = []
    for c in range(K):
        mask = y_seed == c
        x = (z_seed[mask] - mu[c]) / delta[c]
        cov = (x.T @ x) / float(x.shape[0])
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(-eigvals, kind='stable')
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        lam_max = float(eigvals[0])
        if not np.isfinite(lam_max) or lam_max <= 0:
            raise ValueError(f'invalid covariance spectrum for class {c}')
        if int((eigvals > tol_rel * lam_max).sum()) < r:
            raise ValueError(f'class {c} has rank smaller than r={r}')

        blocks, start = [], 0
        for i in range(1, d):
            if eigvals[i - 1] - eigvals[i] >= tol_rel * lam_max:
                blocks.append((start, i))
                start = i
        blocks.append((start, d))

        columns, used = [], 0
        for start, end in blocks:
            columns.append(canonical_block_basis(eigvecs[:, start:end]))
            used += end - start
            if used >= r:
                break
        subspaces.append(np.concatenate(columns, axis=1)[:, :r].copy())
    return np.stack(subspaces, axis=0)


def build_cache(inp, mu, delta, U):
    cache = {}
    for key, z in (('seed', inp.z_seed), ('da', inp.z_da), ('anti', inp.z_anti)):
        cache[key] = metric_features(z, mu, delta, U)
    return cache


def build_priors(r, tail=True):
    m = r * (r + 1) // 2
    theta0 = np.zeros(m, dtype=np.float64)
    theta0[:r] = 1.0
    reg_scale = np.ones(m, dtype=np.float64)
    reg_scale[r:] = 2.0
    if tail:
        theta0 = np.concatenate([theta0, np.array([1.0])])
        reg_scale = np.concatenate([reg_scale, np.array([float(max(int(r), 1))])])
    return theta0, reg_scale


def objective_grad(theta, inp, mu, delta, cache, K, theta0, reg_scale, r,
                   weights, b_anchor=True):
    weights = dict(weights)
    theta = np.asarray(theta, dtype=np.float64)
    r_dim = max(int(r), 1)
    m = theta0.shape[0]
    w = theta[:K * m].reshape(K, m) - theta0[None, :]
    b = theta[K * m:]

    def scores_of(key):
        base, mf = cache[key]
        return b[None, :] - base - np.einsum('km,nkm->nk', w, mf)

    def features_of(key):
        return cache[key][1]

    gw = np.zeros((K, m), dtype=np.float64)
    gb = np.zeros(K, dtype=np.float64)
    total = 0.0

    s_seed = scores_of('seed')
    mf_seed = features_of('seed')

    loss_pos = 0.0
    grad_w = np.zeros((K, m))
    grad_b = np.zeros(K)
    for c in range(K):
        mask = inp.y_seed == c
        score = s_seed[mask, c]
        loss_pos += _sp(-score).mean()
        grad_w[c] += (_sigmoid(-score)[:, None] * mf_seed[mask][:, c, :]).mean(0) / K
        grad_b[c] += -_sigmoid(-score).mean() / K
    loss_pos /= K
    gw += weights['pos'] * grad_w
    gb += weights['pos'] * grad_b
    total += weights['pos'] * loss_pos

    loss_other = 0.0
    grad_w = np.zeros((K, m))
    grad_b = np.zeros(K)
    for c in range(K):
        mask = inp.y_seed == c
        score = s_seed[~mask, c]
        loss_other += _sp(score).mean()
        sig = _sigmoid(score)
        grad_w[c] += -(sig[:, None] * mf_seed[~mask][:, c, :]).mean(0) / K
        grad_b[c] += sig.mean() / K
    loss_other /= K
    gw += weights['other'] * grad_w
    gb += weights['other'] * grad_b
    total += weights['other'] * loss_other

    pos_index = {root: i for i, root in enumerate(inp.seed_identity.tolist())}
    by_class = {}
    for root in sorted(set(inp.da_root.tolist())):
        by_class.setdefault(int(inp.y_seed[pos_index[root]]), []).append(root)

    s_da = scores_of('da')
    mf_da = features_of('da')
    loss_da = 0.0
    grad_w = np.zeros((K, m))
    grad_b = np.zeros(K)
    for c in range(K):
        roots = by_class.get(c, [])
        if not roots:
            continue
        term = 0.0
        accum_w = np.zeros(m)
        accum_b = 0.0
        for root in roots:
            rows = np.where(inp.da_root == root)[0]
            score = s_da[rows, c]
            term += 0.5 * _sp(-score).mean()
            accum_w += (0.5 * _sigmoid(-score)[:, None] * mf_da[rows][:, c, :]).mean(0)
            accum_b += -0.5 * _sigmoid(-score).mean()
        loss_da += term / len(roots)
        grad_w[c] += accum_w / len(roots)
        grad_b[c] += accum_b / len(roots)
    loss_da /= K
    grad_w /= K
    grad_b /= K
    gw += weights['da_sp'] * grad_w
    gb += weights['da_sp'] * grad_b
    total += weights['da_sp'] * loss_da

    wa = float(weights['anti_abs'])
    wr = float(weights['anti_rel'])
    s_anti = scores_of('anti')
    mf_anti = features_of('anti')
    loss_abs = 0.0
    loss_rel = 0.0
    grad_abs_w = np.zeros((K, m))
    grad_rel_w = np.zeros((K, m))
    grad_abs_b = np.zeros(K)
    for c in range(K):
        rows_c = np.where(inp.anti_target == c)[0]
        if len(rows_c) == 0:
            continue
        groups = {}
        for i in rows_c:
            groups.setdefault(str(inp.anti_root[i]), []).append(i)
        abs_w = np.zeros(m)
        rel_w = np.zeros(m)
        abs_b = 0.0
        abs_loss = 0.0
        rel_loss = 0.0
        for root, indices in groups.items():
            indices = np.asarray(indices, dtype=np.int64)
            s_a = s_anti[indices, c]
            s_root = s_seed[pos_index[root], c]
            g1 = _sigmoid(s_a)
            g2 = _sigmoid(1.0 + s_a - s_root)
            mf_a = mf_anti[indices][:, c, :]
            mf_root = mf_seed[pos_index[root]][None, c, :]
            abs_w += -(0.5 * g1[:, None] * mf_a).mean(0)
            rel_w += -(0.5 * g2[:, None] * (mf_a - mf_root)).mean(0)
            abs_b += (0.5 * g1).mean()
            abs_loss += (0.5 * _sp(s_a)).mean()
            rel_loss += (0.5 * _sp(1.0 + s_a - s_root)).mean()
        n_roots = len(groups)
        loss_abs += abs_loss / n_roots
        loss_rel += rel_loss / n_roots
        grad_abs_w[c] += abs_w / n_roots
        grad_rel_w[c] += rel_w / n_roots
        grad_abs_b[c] += abs_b / n_roots
    loss_abs /= K
    loss_rel /= K
    grad_abs_w /= K
    grad_rel_w /= K
    grad_abs_b /= K
    gw += wa * grad_abs_w + wr * grad_rel_w
    gb += wa * grad_abs_b
    total += wa * loss_abs + wr * loss_rel

    radius_prior = float(np.mean((b - 1.0) ** 2)) if b_anchor else \
        float(np.mean((b - b.mean()) ** 2))
    shape_prior = float(np.sum(reg_scale[None, :] * w * w) / (K * r_dim))
    w_shape = float(weights['R_shape'])
    w_radius = float(weights['R_radius'])
    gw += w_shape * 2.0 * reg_scale[None, :] * w / (K * r_dim)
    if b_anchor:
        gb += w_radius * 2.0 * (b - 1.0) / K
    else:
        gb += w_radius * 2.0 * (b - b.mean()) / K
    total += w_shape * shape_prior + w_radius * radius_prior

    return float(total), np.concatenate([gw.ravel(), gb])


def _grad_matrix(packed_grad, r):
    K, _ = packed_grad.shape
    matrix = np.zeros((K, r, r))
    for c in range(K):
        matrix[c][np.arange(r), np.arange(r)] = packed_grad[c][:r]
        k = r
        for i in range(r):
            for j in range(i + 1, r):
                matrix[c][i, j] = matrix[c][j, i] = 0.5 * packed_grad[c][k]
                k += 1
    return matrix


def fit_boundary(inp, config):
    mu, delta = seed_stats(inp.z_seed, inp.y_seed)
    K = int(inp.y_seed.max()) + 1
    r = int(config.rank)
    U = canonical_subspace_perclass(inp.z_seed, inp.y_seed, mu, delta, r=r)
    cache = build_cache(inp, mu, delta, U)
    theta0, reg_scale = build_priors(r, tail=True)
    m = theta0.shape[0]
    m0 = m - 1

    weights = {
        'pos': config.weight_pos,
        'other': config.weight_other,
        'da_sp': config.weight_da,
        'anti_abs': config.weight_anti_abs,
        'anti_rel': config.weight_anti_rel,
        'R_shape': config.weight_shape,
        'R_radius': config.weight_radius,
    }

    def unpack(Hm, b, tail):
        blocks = []
        for c in range(K):
            row = [np.diag(Hm[c]), Hm[c][np.triu_indices(r, 1)],
                   np.array([tail[c]], dtype=np.float64)]
            blocks.append(np.concatenate(row))
        return np.concatenate([np.concatenate(blocks), b])

    def project(Hc):
        eigvals, eigvecs = np.linalg.eigh(0.5 * (Hc + Hc.T))
        lo = 1.0 / config.shape_kappa
        hi = float(config.shape_kappa)
        return (eigvecs * np.clip(eigvals, lo, hi)) @ eigvecs.T

    def project_all(H):
        return np.stack([project(H[c]) for c in range(K)])

    Hm = np.tile(np.eye(r), (K, 1, 1))
    b = np.ones(K)
    tail = np.ones(K)

    J, grad = objective_grad(
        unpack(Hm, b, tail), inp, mu, delta, cache, K, theta0, reg_scale, r, weights)
    J_init = J
    eta = 1.0

    for _ in range(int(config.max_iter)):
        packed_grad = grad[:K * m].reshape(K, m)
        grad_b = grad[K * m:]
        grad_H = _grad_matrix(packed_grad, r)
        grad_tail = packed_grad[:, m0].copy()
        stat = max(
            float(np.linalg.norm(
                (Hm - project_all(Hm - grad_H)).reshape(K, -1), axis=1).max()),
            float(np.abs(b - np.clip(b - grad_b, config.b_min, config.b_max)).max()),
            float(np.abs(tail - np.clip(tail - grad_tail,
                                        config.tail_min, config.tail_max)).max()),
        )
        if stat <= config.tol:
            break

        accepted = False
        for _ in range(61):
            new_H = project_all(Hm - eta * grad_H)
            new_b = np.clip(b - eta * grad_b, config.b_min, config.b_max)
            new_tail = np.clip(tail - eta * grad_tail, config.tail_min, config.tail_max)
            dH = new_H - Hm
            db = new_b - b
            dt = new_tail - tail
            if (float((dH * dH).sum()) == 0.0 and float(np.max(np.abs(db))) == 0.0
                    and float(np.max(np.abs(dt))) == 0.0):
                eta *= 0.5
                continue
            Jc, gc = objective_grad(
                unpack(new_H, new_b, new_tail), inp, mu, delta, cache, K,
                theta0, reg_scale, r, weights)
            rhs = (J + float(np.sum(grad_H * dH)) + float(grad_b @ db)
                   + float(grad_tail @ dt)
                   + (float(np.sum(dH * dH)) + float(db @ db) + float(dt @ dt)) / (2.0 * eta)
                   + 1e-12 * max(1.0, abs(J)))
            if np.isfinite(Jc) and Jc <= rhs:
                Hm, b, tail, J, grad = new_H, new_b, new_tail, Jc, gc
                accepted = True
                eta = min(eta * 2.0, 1e6)
                break
            eta *= 0.5
        if not accepted:
            raise ValueError('backtracking failed')

    packed_grad = grad[:K * m].reshape(K, m)
    grad_b = grad[K * m:]
    grad_H = _grad_matrix(packed_grad, r)
    grad_tail = packed_grad[:, m0].copy()
    stat = max(
        float(np.linalg.norm((Hm - project_all(Hm - grad_H)).reshape(K, -1), axis=1).max()),
        float(np.abs(b - np.clip(b - grad_b, config.b_min, config.b_max)).max()),
        float(np.abs(tail - np.clip(tail - grad_tail,
                                    config.tail_min, config.tail_max)).max()),
    )
    if stat > config.tol:
        raise ValueError(f'KKT residual {stat:.3e} exceeds tol {config.tol:.1e}')
    if J > J_init + 1e-12 * max(1.0, abs(J_init)):
        raise ValueError('objective increased during optimization')

    theta = unpack(Hm, b, tail)[:K * m].copy()
    return theta, b.copy(), {
        'mu': mu,
        'delta': delta,
        'U': U,
        'theta0': theta0,
    }
