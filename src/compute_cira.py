# compute_cira.py
import json
import argparse

def load_map(path):
    data = json.load(open(path))
    # Map data by query ID
    by_id = {str(x["query_id"]): x for x in data}
    return by_id

def cira_metrics(clean_json, wm_json, k=5):
    clean = load_map(clean_json)
    wm    = load_map(wm_json)

    keys = sorted(set(clean.keys()) & set(wm.keys()))
    n = len(keys)
    if n == 0:
        return {"n": 0, "exact@1": 0.0, f"id@{k}": 0.0}

    exact = 0
    overlap_sum = 0.0

    for qid in keys:
        c_ids = clean[qid].get("retrieved_ids", []) or []
        w_ids = wm[qid].get("retrieved_ids", []) or []

        # exact@1
        if c_ids and w_ids and c_ids[0] == w_ids[0]:
            exact += 1

        # id@k (set intersection / k)
        c_top = set(c_ids[:k])
        w_top = set(w_ids[:k])
        overlap = len(c_top & w_top) / float(k)
        overlap_sum += overlap

    return {
        "n": n,
        "exact@1": exact / n,
        f"id@{k}": overlap_sum / n,
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    res = cira_metrics(args.clean, args.wm, args.k)
    print(json.dumps(res, indent=2, ensure_ascii=False))
"""
python src/compute_cira.py \
  --clean /mnt/ssd/TSF/knot-main/output/main_task/trec-covid/clean_main_task.json \
  --wm    /mnt/ssd/TSF/knot-main/output/main_task/trec-covid/wm_main_task.json \
  --k 5

python src/compute_cira.py \
  --clean /mnt/ssd/TSF/knot-main/output/main_task/nfcorpus/clean_main_task.json \
  --wm    /mnt/ssd/TSF/knot-main/output/main_task/nfcorpus/wm_main_task.json \
  --k 5
  
python src/compute_cira.py \
  --clean /mnt/ssd/TSF/knot-main/output/main_task/nq/clean_main_task.json \
  --wm    /mnt/ssd/TSF/knot-main/output/main_task/nq/wm_main_task.json \
  --k 5
  
python src/compute_cira.py \
  --clean /mnt/ssd/TSF/knot-main/output/main_task/hotpotqa/clean_main_task.json \
  --wm    /mnt/ssd/TSF/knot-main/output/main_task/hotpotqa/wm_main_task.json \
  --k 5
  
python src/compute_cira.py \
  --clean /mnt/ssd/TSF/knot-main/output/main_task/msmarco/clean_main_task.json \
  --wm    /mnt/ssd/TSF/knot-main/output/main_task/msmarco/wm_main_task.json \
  --k 5
"""
