"""Source-aware inputs for the ellipsoidal decision boundary."""
import numpy as np


def _sp(x):
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FrontendInputs:
    def __init__(self, z_seed, y_seed, z_da, da_root, z_anti, anti_target,
                 anti_root, head_W, test_z, test_true, seed_identity):
        self.z_seed = np.asarray(z_seed, dtype=np.float64)
        self.y_seed = np.asarray(y_seed, dtype=np.int64)
        self.seed_identity = np.asarray(seed_identity, dtype=object)
        self.z_da = np.asarray(z_da, dtype=np.float64)
        self.da_root = np.asarray(da_root, dtype=object)
        self.z_anti = np.asarray(z_anti, dtype=np.float64)
        self.anti_target = np.asarray(anti_target, dtype=np.int64)
        self.anti_root = np.asarray(anti_root, dtype=object)
        self.head_W = np.asarray(head_W, dtype=np.float64)
        self.test_z = np.asarray(test_z, dtype=np.float64)
        self.test_true = np.asarray(test_true, dtype=np.int64)


def build_inputs(features, num_labels):
    train_index = np.asarray([str(x) for x in features['train_index']])
    train_label = np.asarray(features['train_label'], dtype=np.int64)
    train_feat = np.asarray(features['train_feat'], dtype=np.float64)

    seed_mask = np.array([not x.endswith(('-DA', '-DA-anti')) for x in train_index])
    da_mask = np.array([x.endswith('-DA') for x in train_index])
    anti_mask = np.array([x.endswith('-DA-anti') for x in train_index])

    seed_identity = train_index[seed_mask]
    y_seed = train_label[seed_mask]
    K = int(y_seed.max()) + 1
    if K != num_labels:
        raise ValueError(f'seed label range {K} does not match num_labels {num_labels}')
    if len(set(seed_identity.tolist())) != len(seed_identity):
        raise ValueError('seed identities are not unique')

    pos = {root: i for i, root in enumerate(seed_identity.tolist())}
    da_rows = np.where(da_mask)[0]
    anti_rows = np.where(anti_mask)[0]
    if len(da_rows) == 0 or len(anti_rows) == 0:
        raise ValueError('the enriched training split must contain DA and anti rows')

    da_root = np.array([train_index[i][:-len('-DA')] for i in da_rows], dtype=object)
    anti_root = np.array([train_index[i][:-len('-DA-anti')] for i in anti_rows], dtype=object)
    if not (set(da_root.tolist()) | set(anti_root.tolist())).issubset(pos):
        raise ValueError('a DA/anti row has no matching seed row')

    for i in da_rows:
        root = train_index[i][:-len('-DA')]
        if train_label[i] != y_seed[pos[root]]:
            raise ValueError(f'DA label does not match its seed label: {train_index[i]}')

    offsets = {int(train_label[i]) - int(y_seed[pos[train_index[i][:-len('-DA-anti')]]])
               for i in anti_rows}
    if offsets != {K + 1}:
        raise ValueError(f'anti label offset must be K+1={K + 1}, got {sorted(offsets)}')
    anti_target = train_label[anti_rows] - (K + 1)
    for i, target in zip(anti_rows, anti_target):
        root = train_index[i][:-len('-DA-anti')]
        if int(y_seed[pos[root]]) != int(target):
            raise ValueError(f'anti target does not match its seed class: {train_index[i]}')

    for name, z in (('seed', train_feat[seed_mask]), ('da', train_feat[da_rows]),
                    ('anti', train_feat[anti_rows]), ('test', features['test_feat'])):
        if not np.isfinite(z).all():
            raise ValueError(f'{name} features contain non-finite values')
        err = np.abs(np.linalg.norm(z, axis=1) - 1.0).max()
        if err > 1e-6:
            raise ValueError(f'{name} features are not unit-normalized (max error {err:.2e})')

    if int(np.asarray(features['test_true']).max()) != K:
        raise ValueError(f'open test label must equal K={K}')

    return FrontendInputs(
        z_seed=train_feat[seed_mask],
        y_seed=y_seed,
        z_da=train_feat[da_rows],
        da_root=da_root,
        z_anti=train_feat[anti_rows],
        anti_target=anti_target,
        anti_root=anti_root,
        head_W=features['head_W'],
        test_z=features['test_feat'],
        test_true=features['test_true'],
        seed_identity=seed_identity,
    )


def seed_stats(z_seed, y_seed):
    K = int(y_seed.max()) + 1
    mu = np.zeros((K, z_seed.shape[1]), dtype=np.float64)
    delta = np.zeros(K, dtype=np.float64)
    for c in range(K):
        zc = z_seed[y_seed == c]
        mu[c] = zc.mean(0)
        delta[c] = np.linalg.norm(zc - mu[c], axis=1).mean()
    if not (np.isfinite(mu).all() and np.isfinite(delta).all() and (delta > 0).all()):
        raise ValueError('invalid seed statistics (delta must be strictly positive)')
    return mu, delta


def _D(z, mu, delta):
    t = (z[:, None, :] - mu[None, :, :]) / delta[:, None][None, :, :]
    return t * t


def scores(z, theta, b, mu, delta, U, theta0):
    base, mf = metric_features(z, mu, delta, U)
    k, m = theta.reshape(mu.shape[0], -1).shape
    w = theta.reshape(k, m) - theta0[None, :]
    return b[None, :] - base - np.einsum('km,nkm->nk', w, mf)


def predict(z, theta, b, mu, delta, U, head_W, theta0, K):
    s = scores(z, theta, b, mu, delta, U, theta0)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    wn = head_W / (np.linalg.norm(head_W, axis=1, keepdims=True) + 1e-12)
    c0 = (zn @ wn.T).argmax(1)
    take = s[np.arange(len(z)), c0]
    out = c0.copy()
    out[take < 0] = K
    return out, s


def metric_features(z, mu, delta, U):
    z = np.asarray(z, dtype=np.float64)
    J = z.shape[1]
    r = U.shape[-1]
    D = _D(z, mu, delta)
    base = np.einsum('j,nkj->nk', np.ones(J, dtype=np.float64), D)
    del D
    dl = delta[:, None]
    y = np.einsum('nd,kdr->nkr', z, U)
    muU = np.einsum('kd,kdr->kr', mu, U)
    t = (y - muU[None, :, :]) / dl[None, :, :]

    pairs = [(i, i) for i in range(r)] + \
        [(i, j) for i in range(r) for j in range(i + 1, r)]
    mf = np.empty(t.shape[:2] + (len(pairs),), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        mf[..., idx] = t[..., i] * t[..., j] * (2.0 if i != j else 1.0)

    residual = base - np.einsum('nkj,nkj->nk', t, t)
    mf = np.concatenate([mf, residual[..., None]], axis=-1)
    if not (np.isfinite(base).all() and np.isfinite(mf).all()):
        raise ValueError('non-finite metric features')
    return base, mf
