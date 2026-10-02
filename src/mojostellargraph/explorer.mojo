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
                if deg <= 0:
                    # a node with no neighbours stops the walk, and so does a
                    # non-monotonic `indptr` (the caller may pass its own CSR):
                    # a negative degree is not a column range
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
    """Upstream `naive_weighted_choices(rs, weights, size=None)`.

    ```
    probs = np.cumsum(weights)
    total = probs[-1]
    if total == 0:
        return None
    thresholds = rs.random() if size is None else rs.random(size)
    idx = np.searchsorted(probs, thresholds * total, side="left")
    ```

    `work` is the running-total scratch. A negative weight is upstream's one
    error path; it is reported by returning `-1`. A node with no neighbours has
    no interval to sample from, so it returns `-2` rather than reading
    `colind[lo - 1]`, which is outside the buffer.

    `side="left"` means the first index whose running total is `>= x`, so the
    scan below tests `x <= running` rather than the older v0.8.1 `x < running`.
    The two differ only when a weight is exactly zero, where upstream can
    return the index of that empty sub-interval; the test is kept exact so the
    kernel and `naive_weighted_choices` in Python agree.
    """
    var lo = Int(iget(indptr, node))
    var deg = Int(iget(indptr, node + 1)) - lo
    if deg <= 0:
        return -2
    var total = 0.0
    var k = 0
    while k < deg:
        var w = weights.unsafe_load(k)
        if w < 0.0:
            return -1
        total += w
        work.unsafe_store(k, total)
        k += 1
    if total == 0.0:
        # upstream: `if total == 0: return None`
        return -3
    var x = pcg_uniform(state) * total
    # first index with `work[idx] >= x`
    var idx = 0
    while idx < deg:
        if x <= work.unsafe_load(idx):
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
    edge_weights_addr: Int,
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
    weighted: Int,
):
    """Upstream `BiasedRandomWalk.run`.

    ```
    ip = cast(1.0 / p); iq = cast(1.0 / q)
    for node in nodes:
        for walk_number in range(n):
            walk = [node]
            previous_node = None
            previous_node_neighbours = []
            current_node = node
            for _ in range(length - 1):
                if weighted:
                    neighbours, weights = self.graph.neighbor_arrays(current_node, include_edge_weight=True, use_ilocs=True)
                else:
                    neighbours = self.graph.neighbor_arrays(current_node, use_ilocs=True)
                    weights = np.ones(neighbours.shape, dtype=weight_dtype)
                if len(neighbours) == 0: break
                mask = neighbours == previous_node
                weights[mask] *= ip
                mask |= np.isin(neighbours, previous_node_neighbours)
                weights[~mask] *= iq
                choice = naive_weighted_choices(rs, weights)
                if choice is None: break
                previous_node = current_node
                previous_node_neighbours = neighbours
                current_node = neighbours[choice]
                walk.append(current_node)
            walks.append(walk)
    ```

    A pointer is non-nullable, so `edge_weights_addr` arrives as an `Int` and is
    rebuilt only inside the `weighted != 0` branch that reads it. When
    `weighted == 0` the Python side still passes a live (if unread) buffer,
    because a pointer has no null state here; every transition weight is then
    the constant `1.0` upstream builds with `np.ones(neighbours.shape)`.
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
                    if cdeg <= 0:
                        break

                    # transition_probability over the current node's neighbours
                    k = 0
                    if weighted != 0:
                        var w_ptr = FPtr(unsafe_from_address=edge_weights_addr)
                        while k < cdeg:
                            var nn = Int(iget(colind, clo + k))
                            var weight_cn = w_ptr.unsafe_load(clo + k)
                            if nn == previous_node:
                                work2.unsafe_store(k, ip * weight_cn)
                            elif _contains(indices, deg, nn):
                                work2.unsafe_store(k, 1.0 * weight_cn)
                            else:
                                work2.unsafe_store(k, iq * weight_cn)
                            k += 1
                    else:
                        # `weights = np.ones(neighbours.shape, dtype=weight_dtype)`
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
