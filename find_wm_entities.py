# -*- coding: utf-8 -*-
"""
GROW Watermarking - Core Design v3 (nfcorpus example, with relation-labeled mediators)
- Directed graph (E1,E2,R) ≡ E1->E2
- Global frontier L: entities within centrality percentile [pct_low, pct_high) that allow 2-hop expansion
- Selection: HMAC(key, counter || anchor) only (no dataset_hash, no anchor/target tags)
- Ranking: candidate targets sorted by paths(A,C) in descending order
- Dead-end handling: increment counter and re-sample anchor from L
- One-use principle: each entity can be used once (with optional promotion: target C → next-step anchor)
- Mediators stored in structured format: [{"E2": ..., "R12": [...], "R23": [...]}]
"""

import json
import hmac
import hashlib
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, Set, List, Tuple

# ----------------------------
# Default parameters
# ----------------------------
DEFAULT_KEY = b"swlab"                    # Secret key for HMAC
PCT_LOW_DEFAULT = 0                       # Lower percentile bound for centrality
PCT_HIGH_DEFAULT = 95                     # Upper percentile bound for centrality (exclusive)
MIN_PATHS_DEFAULT = 2                     # Minimum paths(A,C) threshold
BUDGET_LINKS_DEFAULT = 50                 # Maximum number of watermark links to generate
MEDIATOR_SAMPLE_MAX = 12                  # Max number of mediator samples to record

# Policy: allow target → anchor promotion (chain mode)
ALLOW_PROMOTE_TARGET_AS_ANCHOR = True

# ----------------------------
# Utility functions
# ----------------------------
def hmac_u64(key: bytes, msg: bytes) -> int:
    """HMAC-SHA256 → take the top 8 bytes → unsigned 64-bit integer."""
    d = hmac.new(key, msg, hashlib.sha256).digest()
    return int.from_bytes(d[:8], "big", signed=False)

def unbiased_index(key: bytes, anchor: str, n: int, counter_start: int = 0) -> Tuple[int, int]:
    """
    Unbiased index sampling using HMAC(key, counter || anchor).
    - anchor: current anchor entity (string)
    - n: length of candidate list
    - counter_start: initial counter for this attempt
    Returns: (index, next_counter)
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    U64 = 1 << 64
    limit = (U64 // n) * n
    c = counter_start
    while True:
        msg = str(c).encode("utf-8") + b"|" + anchor.encode("utf-8")
        r = hmac_u64(key, msg)
        if r < limit:
            return r % n, c + 1
        c += 1  # rejection, increment counter

def is_relation_token(s: str) -> bool:
    """Heuristic: relation labels are mostly uppercase/underscore."""
    if not isinstance(s, str) or not s:
        return False
    return (any(ch.isalpha() for ch in s)
            and all(ch.isupper() or ch == "_" or ch.isdigit() for ch in s))

# ----------------------------
# Data loading
# ----------------------------
def load_entities(ent_path: Path) -> Dict[str, str]:
    with open(ent_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}

def load_relations(rel_path: Path) -> List[Tuple[str, str, str]]:
    with open(rel_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    triples = []
    for t in raw:
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            continue
        a, b, c = t
        if is_relation_token(b) and not is_relation_token(c):
            e1, r, e2 = a, b, c
        else:
            e1, e2, r = a, b, c
        triples.append((str(e1), str(e2), str(r)))
    return triples

# ----------------------------
# Graph construction
# ----------------------------
def build_graph(triples: List[Tuple[str, str, str]]):
    """
    Returns:
      - out_adj: u -> {v}
      - in_adj : v -> {u}
      - edge_count: (u,v) -> number of relations
      - direct_edges: {(u,v)}
      - edge_rels: (u,v) -> [r1, r2, ...]  relation labels
    """
    out_adj: Dict[str, Set[str]] = defaultdict(set)
    in_adj: Dict[str, Set[str]] = defaultdict(set)
    edge_count: Dict[Tuple[str, str], int] = defaultdict(int)
    edge_rels: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for u, v, r in triples:
        if u == v:
            continue
        out_adj[u].add(v)
        in_adj[v].add(u)
        edge_count[(u, v)] += 1
        edge_rels[(u, v)].append(r)

    direct_edges = set(edge_count.keys())
    return out_adj, in_adj, edge_count, direct_edges, edge_rels

def degree_centrality(out_adj: Dict[str, Set[str]], in_adj: Dict[str, Set[str]]) -> Dict[str, int]:
    """Simple degree centrality = in_degree + out_degree."""
    nodes = set(out_adj.keys()) | set(in_adj.keys())
    cent = {}
    for n in nodes:
        cent[n] = len(out_adj.get(n, ())) + len(in_adj.get(n, ()))
    return cent

def percentile_thresholds(values: List[int], low: int, high: int) -> Tuple[int, int]:
    """Compute [low, high) percentile thresholds from a list of ints."""
    if not values:
        return 0, 0
    arr = sorted(values)
    def at(pct):
        if pct <= 0: return arr[0]
        if pct >= 100: return arr[-1] + 1
        idx = int((len(arr) - 1) * (pct / 100.0))
        return arr[idx]
    return at(low), at(high)

# ----------------------------
# Two-hop frontier & paths
# ----------------------------
def has_twohop(u: str, out_adj, direct_edges) -> bool:
    """Check quickly if u has any valid 2-hop expansion."""
    for v in out_adj.get(u, ()):
        for w in out_adj.get(v, ()):
            if w != u and w != v and (u, w) not in direct_edges:
                return True
    return False

def frontier_paths(u: str,
                   out_adj: Dict[str, Set[str]],
                   edge_count: Dict[Tuple[str, str], int],
                   direct_edges: Set[Tuple[str, str]]) -> Dict[str, int]:
    """
    Compute F(u) and paths(u,·) as {w: path_count}.
    paths(u,w) = Σ_B count(u,B) * count(B,w)
    """
    result: Dict[str, int] = defaultdict(int)
    for v in out_adj.get(u, ()):
        c_uv = edge_count[(u, v)]
        for w in out_adj.get(v, ()):
            if w == u or w == v or (u, w) in direct_edges:
                continue
            c_vw = edge_count[(v, w)]
            result[w] += c_uv * c_vw
    return result

def mediators_for(u: str, w: str,
                  out_adj: Dict[str, Set[str]],
                  edge_rels: Dict[Tuple[str, str], List[str]]) -> List[dict]:
    """
    Return mediators B and their relation labels for u->B->w.
    Example:
      [{"E2": "B1", "R12": ["ASSOCIATED_WITH"], "R23": ["CAUSES","INCREASES_RISK"]}, ...]
    """
    meds: List[dict] = []
    for v in out_adj.get(u, ()):
        if w in out_adj.get(v, ()):
            r12 = edge_rels.get((u, v), [])
            r23 = edge_rels.get((v, w), [])
            meds.append({"E2": v, "R12": list(r12), "R23": list(r23)})
    return meds

# ----------------------------
# Main algorithm (v3, chain mode)
# ----------------------------
def run(
    base_dir: Path,
    key: bytes = DEFAULT_KEY,
    pct_low: int = PCT_LOW_DEFAULT,
    pct_high: int = PCT_HIGH_DEFAULT,
    min_paths: int = MIN_PATHS_DEFAULT,
    budget_links: int = BUDGET_LINKS_DEFAULT,
    out_tsv: Path | None = None,
    out_json: Path | None = None,
):
    ent_path = base_dir / "entities_dict_llm.json"
    rel_path = base_dir / "relation_list_llm.json"

    # Load entities and relations
    entities = load_entities(ent_path)
    triples = load_relations(rel_path)

    # Build graph
    out_adj, in_adj, edge_count, direct_edges, edge_rels = build_graph(triples)

    # Degree centrality & percentile thresholds
    cent = degree_centrality(out_adj, in_adj)
    low_th, high_th = percentile_thresholds(list(cent.values()), pct_low, pct_high)

    def in_band(n: str) -> bool:
        v = cent.get(n, 0)
        return (v >= low_th) and (v < high_th)

    # Global frontier L
    L = [n for n in cent.keys() if in_band(n) and has_twohop(n, out_adj, direct_edges)]
    if not L:
        raise RuntimeError("Global frontier L is empty. Adjust percentile thresholds or conditions.")

    used_anchor: Set[str] = set()
    used_target: Set[str] = set()
    results = []

    total = 0
    counter = 0
    A: str | None = None  # current anchor

    while total < budget_links:
        # Anchor selection
        if A is None:
            tries = 0
            while True:
                idx, counter = unbiased_index(key, "ANCHOR", len(L), counter)
                cand = L[idx]
                if (cand not in used_anchor) and (cand not in used_target):
                    G_full = frontier_paths(cand, out_adj, edge_count, direct_edges)
                    G = {c: p for c, p in G_full.items()
                         if (c not in used_target)
                         and (ALLOW_PROMOTE_TARGET_AS_ANCHOR or (c not in used_anchor))
                         and in_band(c) and (p >= min_paths)}
                    if G:
                        A = cand
                        break
                tries += 1
                if tries > 2000:
                    raise RuntimeError("Anchor selection failed repeatedly. Relax conditions.")

        # Target selection
        G_full = frontier_paths(A, out_adj, edge_count, direct_edges)
        G = {c: p for c, p in G_full.items()
             if (c not in used_target)
             and (ALLOW_PROMOTE_TARGET_AS_ANCHOR or (c not in used_anchor))
             and in_band(c) and (p >= min_paths)}

        if not G:
            A = None
            continue

        ranked = sorted(G.items(), key=lambda kv: kv[1], reverse=True)
        C_list = [c for c, _ in ranked]

        tries_c = 0
        C = None
        while True:
            idx, counter = unbiased_index(key, A, len(C_list), counter)
            cand_c = C_list[idx]
            if cand_c not in used_target and (ALLOW_PROMOTE_TARGET_AS_ANCHOR or (cand_c not in used_anchor)):
                C = cand_c
                break
            tries_c += 1
            if tries_c > 2000:
                C = None
                break

        if C is None:
            A = None
            continue

        # Save result
        p = G[C]
        meds = mediators_for(A, C, out_adj, edge_rels)
        if len(meds) > MEDIATOR_SAMPLE_MAX:
            meds = meds[:MEDIATOR_SAMPLE_MAX]

        results.append({
            "E1": A,
            "E3": C,
            "paths": p,
            "num_mediators": len(meds),
            "mediators": meds,
            "anchor_deg": cent.get(A, 0),
            "target_deg": cent.get(C, 0),
        })

        used_anchor.add(A)
        used_target.add(C)

        # Chain extension: target becomes next anchor
        A = C if ALLOW_PROMOTE_TARGET_AS_ANCHOR else None
        total += 1

    # Save results
    if out_tsv:
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("E1\tE3\tpaths\tnum_mediators\tmediators\tanchor_deg\ttarget_deg\n")
            for r in results:
                meds_str = _format_mediators_tsv(r["mediators"])
                f.write(
                    f"{r['E1']}\t{r['E3']}\t{r['paths']}\t{r['num_mediators']}\t"
                    f"{meds_str}\t{r['anchor_deg']}\t{r['target_deg']}\n"
                )

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Generated WM links: {len(results)} (budget={budget_links})")
    if results:
        print("\nTop 10 preview:")
        for r in results[:10]:
            print(f"- {r['E1']}  ⇒  {r['E3']}  | paths={r['paths']}  mediators={r['num_mediators']}")

# ----------------------------
# TSV serialization helper
# ----------------------------
def _format_mediators_tsv(meds: List[dict]) -> str:
    blocks: List[str] = []
    for m in meds:
        r12 = "|".join(m.get("R12", [])) if m.get("R12") else ""
        r23 = "|".join(m.get("R23", [])) if m.get("R23") else ""
        blocks.append(f"{m['E2']}|R12={r12}|R23={r23}")
    return ";".join(blocks)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        default="./output/wm_prepare/nfcorpus",
                        help="nfcorpus folder (entities_dict_llm.json / relation_list_llm.json required)")
    parser.add_argument("--key", type=str, default=None,
                        help="HMAC key string. Default uses DEFAULT_KEY")
    parser.add_argument("--pct_low", type=int, default=PCT_LOW_DEFAULT)
    parser.add_argument("--pct_high", type=int, default=PCT_HIGH_DEFAULT)
    parser.add_argument("--min_paths", type=int, default=MIN_PATHS_DEFAULT)
    parser.add_argument("--budget", type=int, default=BUDGET_LINKS_DEFAULT)
    parser.add_argument("--out_tsv", type=str, default="./output/wm_prepare/nfcorpus/wm_links_v3.tsv")
    parser.add_argument("--out_json", type=str, default="./output/wm_prepare/nfcorpus/wm_links_v3.json")
    args = parser.parse_args()

    key_bytes = args.key.encode("utf-8") if args.key is not None else DEFAULT_KEY

    run(
        base_dir=Path(args.base_dir),
        key=key_bytes,
        pct_low=args.pct_low,
        pct_high=args.pct_high,
        min_paths=args.min_paths,
        budget_links=args.budget,
        out_tsv=Path(args.out_tsv) if args.out_tsv else None,
        out_json=Path(args.out_json) if args.out_json else None,
    )



"""
python find_wm_entities.py --base_dir ./output/wm_prepare/trec-covid --out_tsv ./output/wm_prepare/trec-covid/wm_links_v3.tsv --out_json ./output/wm_prepare/trec-covid/wm_links_v3.json
python find_wm_entities.py --base_dir ./output/wm_prepare/nfcorpus --out_tsv ./output/wm_prepare/nfcorpus/wm_links_v3.tsv --out_json ./output/wm_prepare/nfcorpus/wm_links_v3.json
python find_wm_entities.py --base_dir ./output/wm_prepare/nq --out_tsv ./output/wm_prepare/nq/wm_links_v3.tsv --out_json ./output/wm_prepare/nq/wm_links_v3.json
python find_wm_entities.py --base_dir ./output/wm_prepare/hotpotqa --out_tsv ./output/wm_prepare/hotpotqa/wm_links_v3.tsv --out_json ./output/wm_prepare/hotpotqa/wm_links_v3.json
python find_wm_entities.py --base_dir ./output/wm_prepare/msmarco --out_tsv ./output/wm_prepare/msmarco/wm_links_v3.tsv --out_json ./output/wm_prepare/msmarco/wm_links_v3.json

"""