#!/usr/bin/env python3
# sample_msmarco_split.py
import argparse, json, os, hashlib, random

def read_queries_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # BEIR style: {"_id": "...", "text": "..."}
            qid = obj.get("_id") or obj.get("qid") or obj.get("id")
            text = obj.get("text", "")
            if qid is None:
                continue
            items.append((str(qid), text))
    return items  # list[(qid, text)]

def read_queries_tsv(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                # some dumps may use one column (id only) — ignore
                continue
            qid, text = parts[0], "\t".join(parts[1:])
            items.append((str(qid), text))
    return items

def read_qrels_ids(path):
    """
    Accept common BEIR/MSMARCO styles:
      - qid \t q0 \t docid \t rel
      - qid \t docid \t rel
    Returns: set(qid)
    """
    keep = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                qid = parts[0]
                keep.add(str(qid))
    return keep

def derive_seed(seed, hmac_key):
    if seed is not None:
        return seed
    if hmac_key:
        digest = hashlib.sha256(hmac_key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)  # 32-bit
    return 42  # default deterministic

def auto_find_queries_file(dataset_dir, split):
    """
    Try the most specific names first, then fall back.
    Returns path or raises FileNotFoundError.
    """
    candidates = [
        os.path.join(dataset_dir, f"queries.{split}.jsonl"),
        os.path.join(dataset_dir, f"queries.{split}.tsv"),
        os.path.join(dataset_dir, "queries.jsonl"),
        os.path.join(dataset_dir, "queries.tsv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No queries file found. Tried:\n- " + "\n- ".join(candidates)
    )

def auto_find_qrels_file(dataset_dir, split):
    """
    Returns a qrels path if found, else None.
    """
    candidates = [
        os.path.join(dataset_dir, "qrels", f"{split}.tsv"),
        os.path.join(dataset_dir, f"qrels.{split}.tsv"),
        os.path.join(dataset_dir, "qrels.tsv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def load_queries_auto(path):
    if path.endswith(".jsonl"):
        return read_queries_jsonl(path)
    if path.endswith(".tsv"):
        return read_queries_tsv(path)
    # last resort: try jsonl then tsv by extension-less path
    try:
        return read_queries_jsonl(path + ".jsonl")
    except Exception:
        return read_queries_tsv(path + ".tsv")

def main():
    ap = argparse.ArgumentParser(description="Sample 10% queries from MSMARCO split (default: train)")
    ap.add_argument("--dataset_dir", required=True, help="Path to datasets/msmarco")
    ap.add_argument("--out_dir", required=True, help="Output dir, e.g., datasets/msmarco_10pct")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="train")
    ap.add_argument("--restrict_qrels", type=str, default=None,
                    help="(optional) qrels TSV to restrict candidates before sampling; "
                         "if omitted, will try to auto-pick qrels for the split if present")
    ap.add_argument("--exact_n", type=int, default=5030, help="target sample size (default 5030 ≈ 10%)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hmac_key", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) locate queries file for the split
    queries_path = auto_find_queries_file(args.dataset_dir, args.split)
    items = load_queries_auto(queries_path)  # list[(qid,text)]

    # 2) restrict by qrels if provided or auto-detected
    qrels_path = args.restrict_qrels
    if qrels_path is None:
        auto_qrels = auto_find_qrels_file(args.dataset_dir, args.split)
        if auto_qrels:
            qrels_path = auto_qrels
    if qrels_path:
        keep = read_qrels_ids(qrels_path)
        before = len(items)
        items = [it for it in items if it[0] in keep]
        after = len(items)
        print(f"[restrict] {before} -> {after} queries using qrels: {qrels_path}")

    # 3) sample
    total = len(items)
    if total == 0:
        raise SystemExit(f"No queries to sample (split={args.split}). Check queries/qrels files.")
    target_n = max(1, min(args.exact_n, total))

    seed = derive_seed(args.seed, args.hmac_key)
    rng = random.Random(seed)
    items_shuf = items[:]
    rng.shuffle(items_shuf)
    subset = items_shuf[:target_n]

    # 4) write outputs
    ids_path   = os.path.join(args.out_dir, f"query_ids_{args.split}_10pct.txt")
    jsonl_path = os.path.join(args.out_dir, f"queries_{args.split}_10pct.jsonl")
    tsv_path   = os.path.join(args.out_dir, f"queries_{args.split}_10pct.tsv")

    with open(ids_path, "w", encoding="utf-8") as f:
        for qid, _ in subset:
            f.write(f"{qid}\n")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for qid, text in subset:
            f.write(json.dumps({"_id": qid, "text": text}, ensure_ascii=False) + "\n")

    with open(tsv_path, "w", encoding="utf-8") as f:
        for qid, text in subset:
            f.write(f"{qid}\t{text}\n")

    print(f"Split: {args.split}")
    print(f"Total candidates: {total}")
    print(f"Sampled: {len(subset)} (seed={seed})")
    print("Wrote:")
    print(" ", ids_path)
    print(" ", jsonl_path)
    print(" ", tsv_path)

if __name__ == "__main__":
    main()

"""
python src/sample_msmarco.py \
  --dataset_dir ./datasets/msmarco \
  --out_dir     ./datasets/msmarco_10pct \
  --split train \
  --exact_n 5030 \
  --hmac_key "YOUR_KEY"
"""