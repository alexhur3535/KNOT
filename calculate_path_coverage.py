# calculate_path_coverage.py
# -*- coding: utf-8 -*-
"""
Sort 2-hop (E1->E3) pairs in nfcorpus by the number of paths,
and print/save the number of unique entities (E1 ∪ E3) covered when including the top cumulative 0,5,10,...,100%.
"""

import json, math, argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List

def is_relation_token(s: str) -> bool:
    return bool(s) and (any(c.isalpha() for c in s) and all(c.isupper() or c == "_" or c.isdigit() for c in s))

def load_relations(path: Path) -> List[Tuple[str, str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    triples = []
    for t in raw:
        if not (isinstance(t, (list, tuple)) and len(t) == 3): continue
        a, b, c = t
        if is_relation_token(b) and not is_relation_token(c):
            e1, r, e2 = a, b, c
        else:
            e1, e2, r = a, b, c
        e1, e2 = str(e1), str(e2)
        if e1 != e2:
            triples.append((e1, e2, str(r)))
    return triples

def build_graph(triples: List[Tuple[str, str, str]]):
    out_adj: Dict[str, Set[str]] = defaultdict(set)
    edge_count: Dict[Tuple[str, str], int] = defaultdict(int)
    nodes: Set[str] = set()
    for u, v, _ in triples:
        out_adj[u].add(v)
        edge_count[(u, v)] += 1
        nodes.add(u); nodes.add(v)
    direct_edges = set(edge_count.keys())
    return nodes, out_adj, edge_count, direct_edges

def compute_twohop_paths(nodes: Set[str], out_adj, edge_count, direct_edges):
    pair_paths: Dict[Tuple[str, str], int] = defaultdict(int)
    for u in nodes:
        for v in out_adj.get(u, ()):
            cuv = edge_count[(u, v)]
            for w in out_adj.get(v, ()):
                if w == u or (u, w) in direct_edges:  # strictly 2-hop only
                    continue
                cvw = edge_count[(v, w)]
                pair_paths[(u, w)] += cuv * cvw
    return {k: v for k, v in pair_paths.items() if v > 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="./output/wm_prepare/nfcorpus")
    ap.add_argument("--out_csv", default="./output/wm_prepare/nfcorpus/wm_paths_coverage_curve_5pct.csv")
    ap.add_argument("--step", type=int, default=5, help="Percentile interval (default: 5)")
    args = ap.parse_args()

    rel_path = Path(args.base_dir) / "relation_list_llm.json"
    triples = load_relations(rel_path)
    nodes, out_adj, edge_count, direct_edges = build_graph(triples)
    print(f"Loaded nodes: {len(nodes):,}, edges: {len(edge_count):,}")

    pair_paths = compute_twohop_paths(nodes, out_adj, edge_count, direct_edges)
    print(f"Two-hop pairs: {len(pair_paths):,}")

    sorted_pairs = sorted(pair_paths.items(), key=lambda kv: kv[1], reverse=True)
    N = len(sorted_pairs)

    step = max(1, min(args.step, 100))
    percentiles = list(range(0, 101, step))
    cutoff_indices = {p: math.ceil(N * (p / 100.0)) for p in percentiles}

    # 0% is always fixed to 0 pairs / 0 entities
    covered_at = {0: 0}
    seen: Set[str] = set()

    # Iterate only over targets excluding 0% and record cumulatively
    targets = iter([p for p in percentiles if p != 0])
    current = next(targets, None)

    for i, ((u, w), _) in enumerate(sorted_pairs, start=1):
        seen.add(u); seen.add(w)
        while current is not None and i >= cutoff_indices[current]:
            covered_at[current] = len(seen)
            current = next(targets, None)

    # Fill missing percentiles (e.g., when N=0)
    for p in percentiles:
        covered_at.setdefault(p, 0)

    # Print
    print("\nPercentile\t#Pairs\tUnique Entities Covered")
    for p in percentiles:
        k = min(cutoff_indices[p], N)
        print(f"{p:>3}%\t\t{k:>6}\t{covered_at[p]:>8}")

    # # Save CSV
    # out_csv = Path(args.out_csv)
    # out_csv.parent.mkdir(parents=True, exist_ok=True)
    # with open(out_csv, "w", encoding="utf-8") as f:
    #     f.write("percentile,num_pairs,unique_entities\n")
    #     for p in percentiles:
    #         k = min(cutoff_indices[p], N)
    #         f.write(f"{p},{k},{covered_at[p]}\n")
    # print(f"\n✅ Saved: {out_csv}")

if __name__ == "__main__":
    main()

"""
Examples:

python calculate_path_coverage.py --base_dir ./output/wm_prepare/trec-covid
python calculate_path_coverage.py --base_dir ./output/wm_prepare/nfcorpus
python calculate_path_coverage.py --base_dir ./output/wm_prepare/nq
python calculate_path_coverage.py --base_dir ./output/wm_prepare/hotpotqa
python calculate_path_coverage.py --base_dir ./output/wm_prepare/msmarco
"""
