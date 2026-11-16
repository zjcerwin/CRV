"""Finite-response calibration, CRV scoring, and batch acquisition."""
import numpy as np

def calibrate(a):
    """Finite-table fit, including all count accumulation, but excluding file parsing."""
    k = int(a['k'])
    workers = int(a['workers'])
    counts = np.bincount(a['cal_worker'] * k * k + a['cal_y'] * k + a['cal_z'],
                         minlength=workers * k * k).reshape(workers, k, k).astype(float)
    support = counts.sum(axis=(1, 2))
    pooled_counts = counts.sum(axis=0)
    pooled = (pooled_counts + 1) / (pooled_counts.sum(axis=1, keepdims=True) + k)
    tau = float(a['tau'])
    confusion = (counts + tau * pooled) / (counts.sum(axis=2, keepdims=True) + tau)
    blend, power = float(a['blend']), float(a['power'])
    if blend != 1. or power != 1.:
        confusion = (blend * confusion + (1 - blend) * pooled) ** power
        confusion /= confusion.sum(axis=2, keepdims=True)
        pooled = pooled ** power
        pooled /= pooled.sum(axis=1, keepdims=True)
    confusion[support < int(a['min_support'])] = pooled
    confusion = np.concatenate((confusion, pooled[None]), axis=0)
    counts_y = np.bincount(a['cal_truths'], minlength=k)
    prior = (counts_y + 1) / (counts_y.sum() + k)
    if float(a['temperature']) != 1.:
        prior = prior ** (1 / float(a['temperature']))
        prior /= prior.sum()
    return prior, confusion

def current_state(a):
    idx, labels = a['source_indices'], a['history_labels']
    ca = a['confusion'][idx[:, 0]]
    cb = a['confusion'][idx[:, 1]]
    row = np.arange(len(idx))
    p = a['prior'][None] * ca[row, :, labels[:, 0]] * cb[row, :, labels[:, 1]]
    p /= p.sum(axis=1, keepdims=True)
    return p, np.argmax(p, axis=1)

def score(policy, a, posterior):
    if policy == 'random':
        return None
    if policy == 'disagreement':
        return (a['history_labels'][:, 0] != a['history_labels'][:, 1]).astype(float)
    if policy == 'risk_only':
        return 1 - posterior.max(axis=1)
    channel = a['confusion'][a['source_indices'][:, 2]]
    if policy == 'plugin_value':
        branches = (posterior[:, :, None] * channel).max(axis=1)
        if int(a['k']) == 2:
            buy = branches.sum(axis=1)
        else:
            # Preserve the left-to-right response sum of the original natural
            # experiments. Tiny positive scores affect ties at saturation.
            buy = np.zeros(len(posterior))
            for z in range(branches.shape[1]):
                buy += branches[:, z]
        return np.maximum(0., buy - posterior.max(axis=1))
    if policy == 'information_gain':
        def ent(p):
            return -(p * np.log(np.clip(p, 1e-300, 1))).sum(axis=1)
        predictive = np.einsum('ni,nij->nj', posterior, channel)
        expected = np.zeros(len(posterior))
        for z in range(posterior.shape[1]):
            branch = posterior * channel[:, :, z] / predictive[:, z, None]
            expected += predictive[:, z] * ent(branch)
        return np.maximum(0., ent(posterior) - expected)
    raise ValueError(policy)

def select(policy, scores, a):
    tie = a['tie_' + policy]
    order = np.argsort(tie, kind='stable') if policy == 'random' else np.lexsort((tie, -scores))
    return order[:int(a['budget'])]

def terminal_update(a, posterior, stop, selected):
    actions = stop.copy()
    c = a['confusion'][a['source_indices'][selected, 2]]
    z = a['candidate_labels'][selected]
    actions[selected] = np.argmax(posterior[selected] * c[np.arange(len(selected)), :, z], axis=1)
    return actions
