import sys, numpy as np
from chain import Shape, generators
M = int(sys.argv[1]) if len(sys.argv) > 1 else 6
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
gens = generators(M)
S = Shape(); S.add(gens[0][1] if gens[0][0]==(1,0) else [g for g in gens if g[0]==(1,0)][0][1], (1,0))
hist = []
for t in range(steps):
    best = None
    for key, es in gens:
        V = S.polygon(np.vstack([S.edges, np.array(es)]))
        e, r, c = Shape.error(V)
        # normalise: prefer small error; tie-break on smaller growth
        if best is None or e < best[0]: best = (e, r, key, es)
    S.add(best[3], best[2])
    hist.append((t, best[1], best[0], best[2]))
for t, r, e, k in hist[::max(1, steps // 25)]:
    print(f"step {t:4d} r={r:8.2f} err={e:6.3f} added {k}")
print('max err for r>20:', max(e for t, r, e, k in hist if r > 20))
print('counts', dict(sorted(S.counts.items())))
