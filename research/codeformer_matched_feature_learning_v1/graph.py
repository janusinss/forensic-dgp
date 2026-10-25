"""Bounded structural autograd reachability; no framework import or queries.

Callers provide actual grad_fn nodes and exact current adapter parameters. This
walks existing next_functions only. It neither invokes autograd nor constructs
or retains a tensor, model, node or backward graph in its returned metadata.
Reachability is not evidence of a nonzero or finite branch gradient.
"""

ADAPTER_PARAMETER_NAMES = (
    'down.weight', 'down.bias', 'depthwise.weight', 'depthwise.bias',
    'up.weight', 'up.bias',
)
MAX_NODES = 65536
MAX_EDGES = 262144
_ABSENT = object()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_adapter_reachability(root, parameters, *, expected_present,
                                 max_nodes=MAX_NODES, max_edges=MAX_EDGES):
    """Require all six current adapter leaves (B), or their absence (A).

    Iterative DFS keeps strong references to every visited node wrapper until
    the bounded walk ends, preventing Python id reuse from merging nodes. Gray
    nodes detect cycles; completed shared subgraphs remain legal. Only an
    AccumulateGrad.variable object identical to a supplied current parameter
    counts. Names/shapes, intermediate requires_grad and other parameter leaves
    cannot establish this identity binding. None is a valid disconnected root
    only when absence is required. No numerical derivative is evaluated.
    """
    _need(type(parameters) is dict and set(parameters) == set(ADAPTER_PARAMETER_NAMES),
          'Exact six named current U96 adapter parameter objects required')
    _need(all(value is not None for value in parameters.values()) and
          len({id(value) for value in parameters.values()}) == 6,
          'Six distinct nonnull current adapter parameter identities required')
    _need(type(expected_present) is bool, 'Explicit adapter graph presence policy required')
    _need(type(max_nodes) is int and 0 < max_nodes <= MAX_NODES and
          type(max_edges) is int and 0 < max_edges <= MAX_EDGES,
          'Positive bounded graph node/edge caps required')
    expected = {id(value): name for name, value in parameters.items()}
    references = {}  # Strong node-wrapper references persist for this whole walk.
    colors = {}
    reached = set()
    accumulate_leaves = 0
    edges = 0
    stack = [] if root is None else [(root, False)]
    while stack:
        node, exiting = stack.pop()
        identity = id(node)
        if exiting:
            _need(colors.get(identity) == 1, 'Malformed graph DFS completion state')
            colors[identity] = 2
            continue
        color = colors.get(identity, 0)
        _need(color != 1, 'Cycle in supplied autograd graph refused')
        if color == 2:
            _need(references[identity] is node, 'Autograd node-wrapper identity changed')
            continue
        _need(len(references) < max_nodes, 'Autograd graph node cap exceeded')
        references[identity] = node
        colors[identity] = 1
        try:
            following = node.next_functions
            variable = getattr(node, 'variable', _ABSENT)
        except Exception as exc:
            raise ValueError('Unreadable autograd node structure') from exc
        _need(type(following) is tuple, 'Autograd next_functions must be an exact tuple')
        _need(edges + len(following) <= max_edges, 'Autograd graph edge cap exceeded')
        edges += len(following)
        if type(node).__name__ == 'AccumulateGrad':
            _need(variable is not _ABSENT and variable is not None and not following,
                  'AccumulateGrad must expose one variable and no outgoing edges')
            accumulate_leaves += 1
            name = expected.get(id(variable))
            if name is not None:
                _need(variable is parameters[name], 'Current adapter leaf identity mismatch')
                reached.add(name)
        else:
            _need(variable is _ABSENT, 'Only AccumulateGrad may advertise a parameter variable')
        stack.append((node, True))
        children = []
        for entry in following:
            _need(type(entry) is tuple and len(entry) == 2 and
                  type(entry[1]) is int and entry[1] >= 0,
                  'Exact autograd next-function node/index pair required')
            child = entry[0]
            if child is not None:
                children.append(child)
        # Immediate children are held strongly too, before future inspection.
        for child in reversed(children):
            stack.append((child, False))
    names = set(parameters)
    missing = names - reached
    _need(not missing if expected_present else not reached,
          'All six current adapter leaves required' if expected_present else
          'Frozen current adapter leaves must be absent')
    return {
        'pass': True, 'expected_present': expected_present,
        'parameter_names': sorted(names), 'reachable_parameter_names': sorted(reached),
        'current_adapter_leaves_reached': len(reached),
        'visited_nodes': len(references), 'next_function_edges': edges,
        'all_accumulate_grad_leaves': accumulate_leaves,
        'maximum_nodes': max_nodes, 'maximum_edges': max_edges,
        'structural_reachability_only': True,
        'nonzero_or_finite_branch_gradient_proven': False,
        'autograd_queries_or_backward_calls': 0,
        'returned_tensor_or_node_references': False,
    }
