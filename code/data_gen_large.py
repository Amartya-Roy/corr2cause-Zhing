#!/usr/bin/env python3
"""
data_gen_large.py - Generate causal NLI data for graphs with n up to 50+ nodes.

Replaces the C++ data_gen.cpp + data_verbalize.py pipeline for large n values.
Uses a sampling-based approach instead of exhaustive enumeration:
  - Samples random DAGs instead of enumerating all 2^(n*(n-1)/2) possibilities
  - Computes CPDAGs via Meek's rules to identify Markov Equivalence Classes (MECs)
  - Samples from each MEC for relation probability estimation
  - Computes d-separation / CI relations efficiently for large n
  - Outputs the same NLI json format consumed by the downstream finetune/eval code

Requirements:
    pip install networkx numpy

Usage:
    # Generate data for n=7..10 (quick test)
    python data_gen_large.py --min_nodes 7 --max_nodes 10 --num_dags 50

    # Generate data for n=7..50 (full run)
    python data_gen_large.py --min_nodes 7 --max_nodes 50 --num_dags 100

    # Generate data for a single n with custom edge probability
    python data_gen_large.py --min_nodes 20 --max_nodes 20 --num_dags 200 --edge_prob 0.15

    # Also compile train/dev/test splits
    python data_gen_large.py --min_nodes 7 --max_nodes 50 --compile_splits
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from itertools import combinations, permutations

import networkx as nx
import numpy as np


# ============================================================================
# Constants
# ============================================================================

RELATION_TYPES = [
    "parent", "non-parent ancestor", "child", "non-child descendant",
    "has_collider", "has_confounder", "mixed_type"
]

# Same hypothesis templates as the original data_verbalize.py (original style)
PROPERTY2HYP_TEMPLATE = {
    "parent": "{node_i} directly causes {node_j}.",
    "non-parent ancestor": "{node_i} causes something else which causes {node_j}.",
    "child": "{node_j} directly causes {node_i}.",
    "non-child descendant": "{node_j} is a cause for {node_i}, but not a direct one.",
    "has_collider": "There exists at least one collider (i.e., common effect) of {node_i} and {node_j}.",
    "has_confounder": "There exists at least one confounder (i.e., common cause) of {node_i} and {node_j}.",
}


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================================
# 1. Random DAG Sampling
# ============================================================================

def sample_random_dag(n, edge_prob=0.3):
    """
    Sample a random DAG on nodes {1, ..., n}.

    Uses a random permutation as the topological order and adds each possible
    forward edge with probability `edge_prob`.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(1, n + 1))
    perm = list(range(1, n + 1))
    random.shuffle(perm)
    order = {node: idx for idx, node in enumerate(perm)}
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and order[i] < order[j] and random.random() < edge_prob:
                G.add_edge(i, j)
    return G


# ============================================================================
# 2. CPDAG Computation (V-structures + Meek's Rules)
# ============================================================================

def get_v_structures(dag):
    """
    Get v-structures as a frozenset of (parent1, child, parent2) triples
    (with parent1 < parent2 for canonical form).
    """
    skeleton = dag.to_undirected()
    v_structs = set()
    for child in dag.nodes():
        parents = list(dag.predecessors(child))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                p1, p2 = parents[i], parents[j]
                if not skeleton.has_edge(p1, p2):
                    v_structs.add((min(p1, p2), child, max(p1, p2)))
    return frozenset(v_structs)


def dag_to_cpdag_status(dag):
    """
    Compute CPDAG edge statuses using V-structure detection + Meek's rules.

    Returns:
        dict mapping (u, v) -> 'compelled' or 'reversible'
    """
    skeleton = dag.to_undirected()
    status = {(u, v): 'reversible' for u, v in dag.edges()}

    # Step 1: V-structure edges are compelled
    for child in dag.nodes():
        parents = list(dag.predecessors(child))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                if not skeleton.has_edge(parents[i], parents[j]):
                    status[(parents[i], child)] = 'compelled'
                    status[(parents[j], child)] = 'compelled'

    # Step 2: Meek's rules until convergence
    changed = True
    while changed:
        changed = False
        for u, v in list(dag.edges()):
            if status[(u, v)] == 'compelled':
                continue

            # R1: exists w such that w->u compelled and w not adjacent to v
            for w in dag.predecessors(u):
                if status.get((w, u)) == 'compelled' and not skeleton.has_edge(w, v):
                    status[(u, v)] = 'compelled'
                    changed = True
                    break
            if status[(u, v)] == 'compelled':
                continue

            # R2: exists w such that u->w compelled and w->v compelled
            for w in dag.successors(u):
                if w != v and status.get((u, w)) == 'compelled':
                    if dag.has_edge(w, v) and status.get((w, v)) == 'compelled':
                        status[(u, v)] = 'compelled'
                        changed = True
                        break
            if status[(u, v)] == 'compelled':
                continue

            # R3: exists w1, w2 undirected neighbors of u in CPDAG,
            #     both with compelled edges to v, and w1 not adj w2
            candidates = []
            for w in dag.predecessors(v):
                if w == u or status.get((w, v)) != 'compelled':
                    continue
                if skeleton.has_edge(u, w):
                    is_undirected = False
                    if dag.has_edge(u, w) and status.get((u, w)) == 'reversible':
                        is_undirected = True
                    elif dag.has_edge(w, u) and status.get((w, u)) == 'reversible':
                        is_undirected = True
                    if is_undirected:
                        candidates.append(w)
            r3 = False
            for ci in range(len(candidates)):
                for cj in range(ci + 1, len(candidates)):
                    if not skeleton.has_edge(candidates[ci], candidates[cj]):
                        r3 = True
                        break
                if r3:
                    break
            if r3:
                status[(u, v)] = 'compelled'
                changed = True

    return status


# ============================================================================
# 3. MEC Sampling
# ============================================================================

def sample_from_mec(dag, cpdag_status, num_samples=20, max_total_attempts=500):
    """
    Sample DAGs from the same Markov Equivalence Class by randomly orienting
    reversible edges via random topological orderings + rejection sampling.
    """
    # Identify undirected (reversible) edge pairs
    reversible_undirected = set()
    for (u, v), s in cpdag_status.items():
        if s == 'reversible':
            reversible_undirected.add((min(u, v), max(u, v)))
    reversible_list = list(reversible_undirected)

    if not reversible_list:
        return [dag]

    original_v_structs = get_v_structures(dag)
    nodes = list(dag.nodes())
    mec_dags = [dag]
    seen = {frozenset(dag.edges())}

    for _ in range(max_total_attempts):
        if len(mec_dags) >= num_samples:
            break

        new_dag = nx.DiGraph()
        new_dag.add_nodes_from(nodes)

        # Add compelled edges as-is
        for (u, v), s in cpdag_status.items():
            if s == 'compelled':
                new_dag.add_edge(u, v)

        # Orient reversible edges via a random topological ordering
        shuffled = list(nodes)
        random.shuffle(shuffled)
        node_rank = {node: i for i, node in enumerate(shuffled)}
        for u, v in reversible_list:
            if node_rank[u] < node_rank[v]:
                new_dag.add_edge(u, v)
            else:
                new_dag.add_edge(v, u)

        edge_key = frozenset(new_dag.edges())
        if edge_key in seen:
            continue
        if not nx.is_directed_acyclic_graph(new_dag):
            continue
        if get_v_structures(new_dag) != original_v_structs:
            continue

        seen.add(edge_key)
        mec_dags.append(new_dag)

    return mec_dags


# ============================================================================
# 4. Conditional Independence Relations
# ============================================================================

def compute_ci_relations(dag, n, max_cond_size=None):
    """
    Compute conditional independence relations using d-separation tests.

    For small n, enumerates all conditioning subsets.  For large n, uses a
    neighbourhood-based strategy to keep computation tractable.
    """
    nodes = list(range(1, n + 1))

    if max_cond_size is None:
        if n <= 8:
            max_cond_size = n - 2          # exhaustive
        elif n <= 15:
            max_cond_size = 3
        elif n <= 25:
            max_cond_size = 2
        else:
            max_cond_size = 1

    ci_relations = []

    for x, y in combinations(nodes, 2):
        other = [z for z in nodes if z != x and z != y]

        if n <= 15:
            # Enumerate all subsets up to max_cond_size
            for size in range(min(max_cond_size + 1, len(other) + 1)):
                for cond in combinations(other, size):
                    try:
                        if nx.d_separated(dag, {x}, {y}, set(cond)):
                            ci_relations.append([[x, y], sorted(list(cond))])
                    except Exception:
                        pass
        else:
            # Neighbourhood-based: only check conditioning sets built from
            # parents/children of x and y
            parents_x = set(dag.predecessors(x)) - {y}
            parents_y = set(dag.predecessors(y)) - {x}
            children_x = set(dag.successors(x)) - {y}
            children_y = set(dag.successors(y)) - {x}
            all_neighbors = parents_x | parents_y | children_x | children_y

            cond_sets = [frozenset()]          # unconditional
            if parents_x:
                cond_sets.append(frozenset(parents_x))
            if parents_y:
                cond_sets.append(frozenset(parents_y))
            combined_parents = parents_x | parents_y
            if combined_parents:
                cond_sets.append(frozenset(combined_parents))

            # Individual neighbours
            for z in all_neighbors:
                cond_sets.append(frozenset([z]))

            # Pairs of neighbours (cap to keep things fast)
            neighbor_list = sorted(all_neighbors)[:8]
            for z1, z2 in combinations(neighbor_list, 2):
                cond_sets.append(frozenset([z1, z2]))

            seen = set()
            for cond in cond_sets:
                cond_clean = frozenset(c for c in cond if c != x and c != y)
                if cond_clean in seen:
                    continue
                seen.add(cond_clean)
                try:
                    if nx.d_separated(dag, {x}, {y}, set(cond_clean)):
                        ci_relations.append([[x, y], sorted(list(cond_clean))])
                except Exception:
                    pass

    return ci_relations


# ============================================================================
# 5. Pairwise Relation Computation
# ============================================================================

def _descendants_excluding(dag, source, excluded_node):
    """BFS for descendants of `source`, never passing through `excluded_node`."""
    visited = set()
    queue = [source]
    while queue:
        node = queue.pop(0)
        for child in dag.successors(node):
            if child != excluded_node and child not in visited:
                visited.add(child)
                queue.append(child)
    return visited


def _ancestors_excluding(dag, source, excluded_node):
    """BFS for ancestors of `source`, never passing through `excluded_node`."""
    visited = set()
    queue = [source]
    while queue:
        node = queue.pop(0)
        for parent in dag.predecessors(node):
            if parent != excluded_node and parent not in visited:
                visited.add(parent)
                queue.append(parent)
    return visited


def compute_pairwise_relations(dag, n):
    """
    Compute the 7 pairwise relation types for every ordered pair (i, j).

    Returns an (n, n, 7) float array where entry [i-1][j-1][k] is 1.0 if
    relation type k holds between nodes i and j, else 0.0.

    Relation types (matching the original C++ code):
      0 - parent:              i -> j (direct edge)
      1 - non-parent ancestor: i is ancestor of j but not parent
      2 - child:               j -> i (direct edge)
      3 - non-child descendant:j is ancestor of i but not parent
      4 - has_collider:        exists common effect reachable independently
      5 - has_confounder:      exists common cause reachable independently
      6 - mixed_type:          (not used in NLI generation, kept for format compat)
    """
    nodes = list(range(1, n + 1))
    relations = np.zeros((n, n, 7), dtype=float)
    edges = set(dag.edges())

    # Pre-compute full descendant/ancestor sets
    descendants = {node: nx.descendants(dag, node) for node in nodes}

    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            ix, jx = i - 1, j - 1

            # parent: i -> j
            if (i, j) in edges:
                relations[ix][jx][0] = 1.0

            # non-parent ancestor: i is ancestor of j but not parent
            if j in descendants[i] and (i, j) not in edges:
                relations[ix][jx][1] = 1.0

            # child: j -> i
            if (j, i) in edges:
                relations[ix][jx][2] = 1.0

            # non-child descendant: j is ancestor of i but not parent
            if i in descendants[j] and (j, i) not in edges:
                relations[ix][jx][3] = 1.0

            # has_collider:  exists node k reachable from i (not via j)
            #                AND reachable from j (not via i)
            desc_i_no_j = _descendants_excluding(dag, i, j)
            desc_j_no_i = _descendants_excluding(dag, j, i)
            if desc_i_no_j & desc_j_no_i:
                relations[ix][jx][4] = 1.0

            # has_confounder: exists node k that can reach i (not via j)
            #                 AND can reach j (not via i)
            anc_i_no_j = _ancestors_excluding(dag, i, j)
            anc_j_no_i = _ancestors_excluding(dag, j, i)
            if anc_i_no_j & anc_j_no_i:
                relations[ix][jx][5] = 1.0

            # mixed_type: not generated as a hypothesis, kept as 0

    return relations


# ============================================================================
# 6. Verbalization
# ============================================================================

def node_ix2surface_form(i, n):
    """
    Convert 1-based node index to variable name.

    For n <= 26 matches the original: 1->Z, 2->Y, 3->X, ...
    For n > 26 uses X1, X2, ...
    """
    if n <= 26:
        return chr(91 - i)   # 1 -> Z, 2 -> Y, 3 -> X, ...
    else:
        return f"X{i}"


def list2text(items):
    """Format a list as 'A, B and C'."""
    items = [str(x) for x in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def generate_nli_for_mec(mec_dags, ci_relations, n, mec_ix):
    """
    Generate NLI data items for a single MEC.

    For each ordered pair (node_i, node_j) with node_j > node_i, and for each
    of the 6 causal relation types, produces one NLI example with:
      - premise:    describes correlations and conditional independencies
      - hypothesis: a causal claim
      - relation:   entailment / contradiction / neutral
      - id:         structured identifier (parsed by downstream eval code)
    """
    # --- Compute relation probabilities across MEC members ---
    rel_sum = np.zeros((n, n, 7))
    for dag in mec_dags:
        rels = compute_pairwise_relations(dag, n)
        rel_sum += rels
    rel_probs = rel_sum / len(mec_dags)

    prob2label = lambda p: (
        'entailment' if p == 1
        else ('contradiction' if p == 0 else 'neutral')
    )

    sf = lambda i: node_ix2surface_form(i, n)
    nodes = list(range(1, n + 1))

    # --- Process CI relations into correlations & conditional independencies ---
    two_nodes2conds = defaultdict(list)
    for pair, cond in ci_relations:
        two_nodes2conds[tuple(pair)].append(cond)
    two_nodes2conds = {k: sorted(v, key=len) for k, v in two_nodes2conds.items()}

    # Correlations = pairs that are NOT unconditionally independent
    corrs = []
    for pair in sorted(combinations(nodes, 2)):
        uncond_ind = (
            pair in two_nodes2conds
            and two_nodes2conds[pair]
            and len(two_nodes2conds[pair][0]) == 0
        )
        if not uncond_ind:
            corrs.append(pair)

    cond_inds = [
        list(k) + [c]
        for k, cs in two_nodes2conds.items()
        for c in cs
    ]
    cond_inds = sorted(cond_inds)

    # --- Build premise ---
    corr_stmts = [f"{sf(i)} correlates with {sf(j)}." for i, j in corrs]
    ci_stmts = [
        (f"{sf(i)} and {sf(j)} are independent given "
         f"{list2text([sf(z) for z in cs])}."
         if cs
         else f"{sf(i)} is independent of {sf(j)}.")
        for i, j, cs in cond_inds
    ]
    all_vars = [sf(i) for i in nodes]

    premise = (
        f"Suppose there is a closed system of {n} variables, "
        f"{list2text(all_vars)}. All the statistical relations among "
        f"these {n} variables are as follows: "
    )
    if corr_stmts:
        premise += ' '.join(corr_stmts)
        if ci_stmts:
            premise += " However, "
    if ci_stmts:
        premise += ' '.join(ci_stmts)

    # --- Generate hypotheses ---
    nli_data = []
    for node_i, node_j in sorted(permutations(nodes, 2)):
        if node_j < node_i:
            continue

        for prop, tmpl in PROPERTY2HYP_TEMPLATE.items():
            ni_s, nj_s = sf(node_i), sf(node_j)

            # Randomised presentation (same logic as original data_verbalize.py)
            if prop in ("has_collider", "has_confounder"):
                if random.random() < 0.5:
                    hyp = tmpl.format(node_i=nj_s, node_j=ni_s)
                else:
                    hyp = tmpl.format(node_i=ni_s, node_j=nj_s)
            elif prop in ("non-parent ancestor", "non-child descendant"):
                if random.random() < 0.5:
                    alt = PROPERTY2HYP_TEMPLATE[
                        "non-child descendant"
                        if prop == "non-parent ancestor"
                        else "non-parent ancestor"
                    ]
                    hyp = alt.format(node_i=nj_s, node_j=ni_s)
                else:
                    hyp = tmpl.format(node_i=ni_s, node_j=nj_s)
            else:
                hyp = tmpl.format(node_i=ni_s, node_j=nj_s)

            rel_ix = RELATION_TYPES.index(prop)
            prob = rel_probs[node_i - 1, node_j - 1, rel_ix]
            label = prob2label(prob)

            nli_data.append({
                'premise': premise,
                'hypothesis': hyp,
                'relation': label,
                'id': (
                    f'num_nodes={n}__mec_id={mec_ix}__'
                    f'node_i={node_i}__node_j={node_j}__'
                    f'causal_relation={prop.replace(" ", "_")}__'
                    f'prob={prob:.2f}'
                ),
            })

    return nli_data


# ============================================================================
# 7. Main Pipeline
# ============================================================================

def generate_data_for_n(n, num_dags=100, num_mec_samples=20,
                        edge_prob=None, seed=0):
    """Generate all NLI data for a given number of nodes."""
    set_seed(seed)

    if edge_prob is None:
        # Keep average degree ~ 3-4 regardless of n
        edge_prob = min(0.5, 6.0 / max(n - 1, 1))

    print(f"\n{'=' * 60}")
    print(f"  Generating data for n = {n} nodes")
    print(f"  num_dags={num_dags}, edge_prob={edge_prob:.3f}, "
          f"mec_samples={num_mec_samples}")
    print(f"{'=' * 60}")

    all_nli = []
    mec_ix = 0
    seen_mecs = set()
    t0 = time.time()

    for dag_idx in range(num_dags):
        if (dag_idx + 1) % max(1, num_dags // 10) == 0 or dag_idx == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:6.1f}s] DAG {dag_idx + 1:>4}/{num_dags}, "
                  f"unique MECs so far: {mec_ix}")

        dag = sample_random_dag(n, edge_prob)

        # De-duplicate MECs (same skeleton + same v-structures = same MEC)
        skeleton_key = frozenset(
            (min(u, v), max(u, v)) for u, v in dag.edges()
        )
        v_struct_key = get_v_structures(dag)
        mec_key = (skeleton_key, v_struct_key)
        if mec_key in seen_mecs:
            continue
        seen_mecs.add(mec_key)

        # Compute CPDAG
        cpdag_status = dag_to_cpdag_status(dag)

        # Sample additional DAGs from the same MEC
        mec_dags = sample_from_mec(
            dag, cpdag_status, num_samples=num_mec_samples
        )

        # Compute CI relations (all MEC members share the same CI structure)
        ci_rels = compute_ci_relations(dag, n)

        # Generate NLI examples
        nli = generate_nli_for_mec(mec_dags, ci_rels, n, mec_ix)
        all_nli.extend(nli)
        mec_ix += 1

    elapsed = time.time() - t0
    print(f"  [{elapsed:6.1f}s] Done: {len(all_nli)} NLI samples "
          f"from {mec_ix} unique MECs\n")

    # Shuffle (same as original pipeline)
    random.shuffle(all_nli)
    return all_nli


def num_samples2splits(num_samples):
    """Same split logic as the original data_stats.py."""
    num_test = min(1000, num_samples // 10)
    num_dev = num_test
    if num_samples < 1000:
        num_test = num_samples // 2
        num_dev = num_samples - num_test
    return {
        'test': num_test,
        'dev': num_dev,
        'train': num_samples - num_test - num_dev,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate causal NLI data for graphs with n up to 50+ nodes'
    )
    parser.add_argument(
        '--min_nodes', type=int, default=7,
        help='Minimum number of nodes (default: 7)')
    parser.add_argument(
        '--max_nodes', type=int, default=50,
        help='Maximum number of nodes (default: 50)')
    parser.add_argument(
        '--num_dags', type=int, default=100,
        help='Number of random DAGs to sample per n (default: 100)')
    parser.add_argument(
        '--num_mec_samples', type=int, default=20,
        help='DAGs to sample from each MEC for probability estimation '
             '(default: 20)')
    parser.add_argument(
        '--edge_prob', type=float, default=None,
        help='Edge probability (default: auto, ~6/(n-1) to keep avg degree ~3)')
    parser.add_argument(
        '--output_dir', type=str, default='../data',
        help='Output directory (default: ../data)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed (default: 0)')
    parser.add_argument(
        '--compile_splits', action='store_true',
        help='Also compile train/dev/test splits')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_t0 = time.time()
    for n in range(args.min_nodes, args.max_nodes + 1):
        nli_data = generate_data_for_n(
            n,
            num_dags=args.num_dags,
            num_mec_samples=args.num_mec_samples,
            edge_prob=args.edge_prob,
            seed=args.seed,
        )

        outfile = os.path.join(args.output_dir, f'causalnli_{n}nodes.json')
        # Use compact JSON for large n to avoid multi-GB files
        indent = 2 if n <= 15 else None
        with open(outfile, 'w') as f:
            json.dump(nli_data, f, indent=indent)
        size_mb = os.path.getsize(outfile) / 1e6
        print(f"[Info] Saved {len(nli_data)} samples ({size_mb:.1f} MB) "
              f"-> {outfile}")

    total_elapsed = time.time() - total_t0
    print(f"\n[Info] Total time: {total_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Optional: compile train / dev / test splits
    # ------------------------------------------------------------------
    if args.compile_splits:
        print(f"\n{'=' * 60}")
        print("  Compiling train / dev / test splits")
        print(f"{'=' * 60}")

        split2data = defaultdict(list)

        for n in range(args.min_nodes, args.max_nodes + 1):
            infile = os.path.join(args.output_dir, f'causalnli_{n}nodes.json')
            if not os.path.exists(infile):
                continue
            with open(infile) as f:
                data = json.load(f)

            splits = num_samples2splits(len(data))
            idx = 0
            for split_name, size in splits.items():
                split2data[split_name].extend(data[idx:idx + size])
                idx += size

        split_dir = os.path.join(args.output_dir, 'data_3class_from_Z')
        os.makedirs(split_dir, exist_ok=True)

        for split_name, data in split2data.items():
            outfile = os.path.join(split_dir, f'{split_name}.json')
            with open(outfile, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  {split_name:>5}: {len(data):>8} samples -> {outfile}")


if __name__ == '__main__':
    main()
