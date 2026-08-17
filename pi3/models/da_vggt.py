"""DA-VGGT appearance/pose-aware partitioning for Pi3."""
from __future__ import annotations
import numpy as np

def cosine_similarity(features):
    x = features.float().numpy(); x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return np.clip(x @ x.T, -1.0, 1.0)

def diversity_partition(sim, chunk_size, anchors=(0,), iters=5):
    n = len(sim); groups = [[] for _ in range((n + chunk_size - 1) // chunk_size)]
    cap = max(1, (n + len(groups) - 1) // len(groups)); score = 1.0 - sim; np.fill_diagonal(score, 0.)
    remaining = set(range(n)) - set(anchors)
    for g in groups:
        while remaining and len(g) < cap:
            _, pick = max(((score[i, g].sum() if g else score[i].sum(), i) for i in remaining)); g.append(pick); remaining.remove(pick)
    for i in remaining: min(groups, key=len).append(i)
    for _ in range(iters):
        changed = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                best = (0., None, None)
                for ia, x in enumerate(groups[a]):
                    for ib, y in enumerate(groups[b]):
                        gain = score[y, groups[a]].sum() + score[x, groups[b]].sum() - score[x, groups[a]].sum() - score[y, groups[b]].sum()
                        if gain > best[0]: best = (gain, ia, ib)
                if best[1] is not None:
                    ia, ib = best[1:]; groups[a][ia], groups[b][ib] = groups[b][ib], groups[a][ia]; changed = True
        if not changed: break
    anchor_set = set(anchors)
    return [[anchors[0], *[x for x in g if x not in anchor_set]] for g in groups]

def pseudo_positions(sim, indices, positions, gamma):
    result = np.zeros((len(sim), 3)); known = np.asarray(indices); result[known] = positions; known_set = set(indices)
    for i in range(len(sim)):
        if i in known_set: continue
        logits = sim[i, known] / max(gamma, 1e-6); logits -= logits.max(); w = np.exp(logits); w /= w.sum(); result[i] = w @ positions
    return result

def pose_weighted_similarity(sim, pseudo, tau=None):
    dist = np.linalg.norm(pseudo[:, None] - pseudo[None, :], axis=-1)
    tau = float(np.median(dist[dist > 0])) if tau is None and np.any(dist > 0) else (tau or 1.)
    return sim * np.exp(-dist / max(tau, 1e-6))
