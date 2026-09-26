"""Port of `stellargraph/data/explorer.py` (v0.8.1): `UniformRandomWalk.run`,
`BiasedRandomWalk.run` and `naive_weighted_choices`.

Upstream draws from `numpy.random.RandomState`, which cannot cross the C ABI
and whose stream is not reproducible outside NumPy. The LCG below is the
documented replacement: the walk logic, the neighbour shuffling, the dead-end
`break`, and the `1/p` / `1.0` / `1/q` transition weights are upstream's, and
the generator is fixed so the same seed gives the same walks on both sides of
the FFI.

`walks_out` is `[n_root * n, length]` of node indices with `-1` padding, plus a
`lens_out` vector of the actual walk lengths: a list of variable-length lists
does not survive the C ABI.
"""

from mojostellargraph.types import FPtr, IPtr, iget, iput

comptime PCG_MULT = 6364136223846793005
comptime PCG_INCR = 1442695040888963407


def pcg_next(state: UInt64) -> UInt64:
    return state * PCG_MULT + PCG_INCR


def pcg_bounded(state: UInt64, bound: Int) -> Int:
    """Uniform integer in `[0, bound)`, rejection sampled so there is no bias."""
    var s = state
    if bound <= 1:
        return 0
    # Rejection sample the top 32 bits against the largest multiple of `bound`
    # below 2**32, so the draw is uniform and the loop always terminates.
    var threshold = (0x100000000 // UInt64(bound)) * UInt64(bound)
    while True:
        s = pcg_next(s)
        if (s >> 32) < threshold:
            return Int((s >> 32) % UInt64(bound))
    return 0


def pcg_uniform(state: UInt64) -> Float64:
    """`random.random()`: a float in `[0, 1)` from the top 53 bits."""
    var s = pcg_next(state)
    return Float64(s >> 11) * (1.0 / 9007199254740992.0)


def shuffle(items: IPtr, m: Int, state: UInt64) -> UInt64:
    """`rs.shuffle(neighbours)` in upstream's uniform walk: Fisher-Yates.

    The advanced state is returned because the C ABI cannot return a tuple.
    """
    var s = state
    var i = m - 1
    while i > 0:
        var j = pcg_bounded(s, i + 1)
        s = pcg_next(s)
        var t = items.unsafe_load(i)
        items.unsafe_store(i, items.unsafe_load(j))
        items.unsafe_store(j, t)
        i -= 1
    return s


def uniform_random_walk(
    indptr: IPtr,
    colind: IPtr,
    root_nodes: IPtr,
    walks_out: IPtr,
    lens_out: IPtr,
    work: IPtr,
    n: Int,
    n_walks: Int,
    length: Int,
    seed: Int,
):
    """Upstream `UniformRandomWalk.run(nodes, n, length, seed=None)`.

    ```
    walks = []
    for node in nodes:
        for walk_number in range(n):
            walk = list()
            current_node = node
            for _ in range(length):
                walk.extend([current_node])
                neighbours = self.neighbors(current_node)
                if len(neighbours) == 0:
                    break
                else:
                    rs.shuffle(neighbours)
                    current_node = neighbours[0]
            walks.append(walk)
    return walks
    """
    var state = UInt64(seed) + PCG_INCR
    var w = 0
    for r in range(n):
        var node = Int(iget(root_nodes, r))
        for _ in range(n_walks):
            var base = w * length
            var steps = 0
            var current_node = node
            for _ in range(length):
                iput(walks_out, base + steps, current_node)
                steps += 1
                var lo = Int(iget(indptr, current_node))
                var deg = Int(iget(indptr, current_node + 1)) - lo
                if deg == 0:
                    # for whatever reason this node has no neighbours so stop
                    break
                var k = 0
                while k < deg:
                    iput(work, k, Int(iget(colind, lo + k)))
                    k += 1
                state = shuffle(work, deg, state)
                current_node = iget(work, 0)
            iput(lens_out, w, steps)
            var pad = steps
            while pad < length:
                iput(walks_out, base + pad, -1)
                pad += 1
            w += 1


def naive_weighted_choices(
    indptr: IPtr,
    colind: IPtr,
    weights: FPtr,
    work: FPtr,
    node: Int,
    state: UInt64,
) -> Int:
    """Upstream `naive_weighted_choices(rs, weights)`.

    ```
    subinterval_ends = []
    running_total = 0
    for w in weights:
        if w < 0: raise ValueError(...)
        running_total += w
        subinterval_ends.append(running_total)
    x = rs.random() * running_total
    for idx, end in enumerate(subinterval_ends):
        if x < end: break
    return idx
    ```

    `work` is the running-total scratch. A negative weight is upstream's one
    error path; it is reported by returning `-1`.
    """
    var total = 0.0
    var k = 0
    var lo = Int(iget(indptr, node))
    var deg = Int(iget(indptr, node + 1)) - lo
    while k < deg:
        var w = weights.unsafe_load(k)
        if w < 0.0:
            return -1
        total += w
        work.unsafe_store(k, total)
        k += 1
    var x = pcg_uniform(state) * total
    var idx = 0
    while idx < deg:
        if x < work.unsafe_load(idx):
            break
        idx += 1
    if idx >= deg:
        idx = deg - 1
    return Int(iget(colind, lo + idx))


def _contains(items: IPtr, m: Int, value: Int) -> Bool:
    for i in range(m):
        if Int(items.unsafe_load(i)) == value:
            return True
    return False


def biased_random_walk(
    indptr: IPtr,
    colind: IPtr,
    root_nodes: IPtr,
    walks_out: IPtr,
    lens_out: IPtr,
    work: FPtr,
    work2: FPtr,
    indices: IPtr,
    n: Int,
    n_walks: Int,
    length: Int,
    p: Float64,
    q: Float64,
    seed: Int,
):
    """Upstream `BiasedRandomWalk.run(nodes, n, p=1.0, q=1.0, length, seed)`.

    ```
    ip = 1.0 / p
    iq = 1.0 / q
    for node in nodes:
        for walk_number in range(n):
            walk = [node]
            neighbours = self.neighbors(node)
            previous_node = node
            previous_node_neighbours = neighbours

            def transition_probability(nn, current_node, weighted, edge_weight_label):
                weight_cn = 1.0                       # unweighted walk
                if nn == previous_node:               # d_tx = 0
                    return ip * weight_cn
                elif nn in previous_node_neighbours:  # d_tx = 1
                    return 1.0 * weight_cn
                else:                                 # d_tx = 2
                    return iq * weight_cn

            if neighbours:
                current_node = rs.choice(neighbours)
                for _ in range(length - 1):
                    walk.append(current_node)
                    neighbours = self.neighbors(current_node)
                    if not neighbours:
                        break
                    choice = naive_weighted_choices(rs, transition_probability(nn, ...))
                    previous_node = current_node
                    previous_node_neighbours = neighbours
                    current_node = neighbours[choice]
            walks.append(walk)
    ```

    Only the unweighted walk is ported: upstream's `weighted=True` reads a
    per-edge label off a `networkx` graph, which this port has no graph object
    for. See the README.
    """
    var state = UInt64(seed) + PCG_INCR
    var ip = 1.0 / p
    var iq = 1.0 / q
    var w = 0
    for r in range(n):
        var node = Int(iget(root_nodes, r))
        for _ in range(n_walks):
            var base = w * length
            var steps = 0
            iput(walks_out, base, node)
            steps += 1
            var current_node = node
            var previous_node = node

            var lo = Int(iget(indptr, node))
            var deg = Int(iget(indptr, node + 1)) - lo
            if deg > 0:
                # previous_node_neighbours = neighbours
                var k = 0
                while k < deg:
                    iput(indices, k, Int(iget(colind, lo + k)))
                    k += 1
                current_node = Int(iget(colind, lo + pcg_bounded(state, deg)))
                state = pcg_next(state)

                for _ in range(length - 1):
                    iput(walks_out, base + steps, current_node)
                    steps += 1
                    var clo = Int(iget(indptr, current_node))
                    var cdeg = Int(iget(indptr, current_node + 1)) - clo
                    if cdeg == 0:
                        break

                    # transition_probability over the current node's neighbours
                    k = 0
                    while k < cdeg:
                        var nn = Int(iget(colind, clo + k))
                        if nn == previous_node:
                            work2.unsafe_store(k, ip)
                        elif _contains(indices, deg, nn):
                            work2.unsafe_store(k, 1.0)
                        else:
                            work2.unsafe_store(k, iq)
                        k += 1
                    # `naive_weighted_choices` draws `rs.random()` internally,
                    # so the caller's advance is the second of the two steps
                    var choice = naive_weighted_choices(
                        indptr, colind, work2, work, current_node, state
                    )
                    state = pcg_next(pcg_next(state))
                    if choice < 0:
                        break
                    previous_node = current_node
                    # previous_node_neighbours = neighbours
                    k = 0
                    while k < cdeg:
                        iput(indices, k, Int(iget(colind, clo + k)))
                        k += 1
                    deg = cdeg
                    current_node = choice
            iput(lens_out, w, steps)
            var pad = steps
            while pad < length:
                iput(walks_out, base + pad, -1)
                pad += 1
            w += 1
