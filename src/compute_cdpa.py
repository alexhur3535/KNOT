# src/compute_cdpa.py
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import argparse
import time
import random
from typing import Dict, List, Tuple
from utils import load_json, save_json, Log
from models import create_model

# ---------------- Common helpers ----------------
def _build_index(items: List[Dict], key: str = "id") -> Dict[str, Dict]:
    idx = {}
    for it in items:
        k = str(it.get(key, len(idx)))
        idx[k] = it
    return idx

def _pair_ids(baseline: List[Dict], variant: List[Dict], id_key: str, strict_ids: bool):
    bidx = _build_index(baseline, id_key)
    vidx = _build_index(variant,  id_key)
    shared = sorted(set(bidx.keys()) & set(vidx.keys()))
    if not shared and not strict_ids:
        n = min(len(baseline), len(variant))
        shared = [str(i) for i in range(n)]
        bidx = {str(i): baseline[i] for i in range(n)}
        vidx = {str(i): variant[i]  for i in range(n)}
    return bidx, vidx, shared

def _subset_ids(shared_ids: List[str], limit: int, sample: int, seed: int) -> List[str]:
    ids = shared_ids[:]
    if limit > 0:
        ids = ids[:limit]
    if sample > 0 and sample < len(ids):
        random.seed(seed)
        ids = random.sample(ids, sample)
    return ids

# ---------------- LLM-as-Judge prompt ----------------
JUDGE_PROMPT = """
Given two sentences, determine if they convey the same meaning. If they are similar in meaning, return ’yes’; otherwise, return ’no’.
The following situations are also considered as the two sentences expressing the same meaning:
1. One sentence includes the meaning expressed in the other sentence.
2. The two sentences express the same central idea but in different ways.
Sentence 1: {s1}
Sentence 2: {s2}
Output: 'yes' or 'no' only. No explanations, no extra text."""



def _normalize_judge_output(text: str) -> str:
    t = (text or "").strip().strip('"').strip("'")
    t = t.split()[0] if t else t
    t_low = t.lower()
    if t_low.startswith("yes"): return "yes"
    if t_low.startswith("no"):  return "no"
    return t_low

def judge_equivalent(llm, s1: str, s2: str, retries: int = 1) -> Tuple[int, str]:
    last = ""
    for _ in range(max(1, retries + 1)):
        try:
            out = llm.query(JUDGE_PROMPT.format(s1=s1 or "", s2=s2 or ""))
            raw = _normalize_judge_output(str(out))
            last = raw
            if raw in ("yes", "no"):
                return (1 if raw == "yes" else 0), raw
        except Exception as e:
            last = f"error:{e}"
    return 2, last  # unknown

# ---------------- Lightweight progress reporter ----------------
class Progress:
    def __init__(self, total: int, log_interval: int = 50, logger=None):
        self.total = max(1, total)
        self.log_interval = max(1, log_interval)
        self.logger = logger
        self.start = time.time()
        self.last_log_t = self.start

    def _emit(self, msg: str):
        # 로그도 남기고, stdout에도 한 줄 남겨 nohup에서도 보이게
        if self.logger:
            self.logger.info(msg)
        print(msg, flush=True)

    def step(self, done: int, *, yes: int, no: int, unk: int):
        now = time.time()
        if done == self.total or done % self.log_interval == 0 or (now - self.last_log_t) >= 30:
            elapsed = now - self.start
            rate = done / elapsed if elapsed > 0 else 0.0
            remain = (self.total - done) / rate if rate > 0 else float('inf')
            pct = 100.0 * done / self.total
            msg = (f"[CDPA] {done}/{self.total} ({pct:5.1f}%) | "
                   f"yes={yes} no={no} unk={unk} | "
                   f"{rate:.2f} it/s | ETA {remain:6.1f}s")
            self._emit(msg)
            self.last_log_t = now

    def finalize(self, yes: int, no: int, unk: int):
        elapsed = time.time() - self.start
        msg = (f"[CDPA] DONE in {elapsed:.1f}s | "
               f"yes={yes} no={no} unk={unk} | total={self.total}")
        self._emit(msg)

# ---------------- Core: CDPA only ----------------
def compute_cdpa_only(
    clean_json: str,
    wm_json: str,
    out_dir: str,
    *,
    id_key: str = "id",
    text_key: str = "llm_text",
    judge_model: str = "gpt3.5",
    judge_config: str = None,
    limit: int = 0,
    sample: int = 0,
    seed: int = 633,
    strict_ids: bool = False,
    retries: int = 1,
    logger=None,
    log_interval: int = 50
) -> Dict:
    clean = load_json(clean_json)
    wm    = load_json(wm_json)

    # load judge llm
    cfg = judge_config or f"model_configs/{judge_model}_config.json"
    judge_llm = create_model(cfg)
    if logger: logger.info(f"judge model cfg: {cfg}")

    bidx, vidx, shared = _pair_ids(clean, wm, id_key, strict_ids)
    shared = _subset_ids(shared, limit, sample, seed)

    yes = no = unk = 0
    details = []
    total = len(shared)
    prog = Progress(total=total, log_interval=log_interval, logger=logger)

    for i, qid in enumerate(shared, start=1):
        s1 = str(bidx[qid].get(text_key, "")).strip()
        s2 = str(vidx[qid].get(text_key, "")).strip()
        flag, raw = judge_equivalent(judge_llm, s1, s2, retries=retries)
        if flag == 1: yes += 1
        elif flag == 0: no += 1
        else: unk += 1
        details.append({
            "id": qid,
            "baseline_text": s1,
            "variant_text":  s2,
            "judge_raw": raw,
            "flag": flag
        })
        prog.step(i, yes=yes, no=no, unk=unk)

    prog.finalize(yes=yes, no=no, unk=unk)

    score = (yes / total) if total > 0 else 0.0

    os.makedirs(out_dir, exist_ok=True)
    result = {
        "metric": "CDPA",
        "total": total, "yes": yes, "no": no, "unknown": unk,
        "alignment": score,
        "meta": {
            "clean_json": clean_json,
            "wm_json": wm_json,
            "id_key": id_key,
            "text_key": text_key,
            "timestamp": int(time.time())
        },
        "details": details
    }
    save_json(result, os.path.join(out_dir, "cdpa_result.json"))
    save_json({"cdpa_alignment": score, "cdpa_total": total},
              os.path.join(out_dir, "summary.json"))
    return result

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("Compute CDPA from two JSONs containing 'llm_text'")
    ap.add_argument("--clean_json", required=True)
    ap.add_argument("--wm_json",    required=True)
    ap.add_argument("--out_dir",    required=True)

    ap.add_argument("--judge_model", default="gpt3.5",
                    choices=['gpt3.5','claude','gemini','llama','vicuna','mistral'])
    ap.add_argument("--judge_config", default=None)

    ap.add_argument("--id_key",   default="id")
    ap.add_argument("--text_key", default="llm_text")
    ap.add_argument("--limit",  type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed",   type=int, default=633)
    ap.add_argument("--strict_ids", action="store_true")
    ap.add_argument("--retries", type=int, default=1)

    # logging / progress options
    ap.add_argument("--log_file", default=None)
    ap.add_argument("--log_interval", type=int, default=50,
                    help="Print progress every N items (default: 50)")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = args.log_file or os.path.join(args.out_dir, "cdpa.log")
    logger = Log(log_file=log_path).get(__file__)
    logger.info(f"args: {vars(args)}")

    res = compute_cdpa_only(
        clean_json=args.clean_json,
        wm_json=args.wm_json,
        out_dir=args.out_dir,
        id_key=args.id_key,
        text_key=args.text_key,
        judge_model=args.judge_model,
        judge_config=args.judge_config,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
        strict_ids=args.strict_ids,
        retries=args.retries,
        logger=logger,
        log_interval=args.log_interval,
    )

    print(f"CDPA alignment = {res['alignment']:.4f}  (yes/total = {res['yes']}/{res['total']})", flush=True)


"""
python src/compute_cdpa.py \
  --clean_json /mnt/ssd/TSF/grow/output/main_task/trec-covid/clean_main_task.json \
  --wm_json    /mnt/ssd/TSF/grow/output/main_task/trec-covid/wm_main_task.json \
  --out_dir    /mnt/ssd/TSF/grow/output/main_task/trec-covid \
  --judge_model gpt3.5 \
  --retries 1

python src/compute_cdpa.py \
  --clean_json /mnt/ssd/TSF/grow/output/main_task/nfcorpus/clean_main_task.json \
  --wm_json    /mnt/ssd/TSF/grow/output/main_task/nfcorpus/wm_main_task.json \
  --out_dir    /mnt/ssd/TSF/grow/output/main_task/nfcorpus \
  --judge_model gpt3.5 \
  --retries 1
"""