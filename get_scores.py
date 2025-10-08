# get_score.py  (imperceptibility-removed version)
import sys
sys.path.append('/mnt/ssd/TSF/knot-main')  # adjust if needed

import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

from src.utils import load_json
from src.models import GPT  # internal wrapper assuming llm.query(prompt) -> str

# -----------------------------
# Prompt (rubric · JSON only)
# -----------------------------
PROMPT_TEMPLATE = """System:
You are an impartial, deterministic judge of sentence quality.
Follow the rubric strictly and return JSON only. No extra text.

User:
Rate the sentence on a 1–10 integer scale for each criterion.
Evaluate the sentence itself (no web lookup). Penalize obvious watermark artifacts
(e.g., repetitive n-grams, templated phrasing, forced token choices).

Criteria (weights in parentheses; total is weighted sum → 1–10):
1) Fluency/Naturalness (0.25) – grammar and readability.
2) Clarity/Information (0.25) – specificity and meaning.
3) Domain Appropriateness (0.25) – terminology fits the domain.
4) Plausibility/Consistency (0.25) – no obvious contradictions or category errors.

Rules:
- Integers only (1–10) per criterion.
- Keep each rationale short (≤1 sentence).
- Output exactly one JSON object with these keys.

Output JSON schema:
{
  "fluency": int,
  "clarity": int,
  "domain": int,
  "plausibility": int,
  "total_weighted": float,
  "notes": "one-sentence overall remark"
}

Sentence: "{sentence}"
"""

# === Weights: keep only 4 criteria ===
WEIGHTS = {"fluency": 0.25, "clarity": 0.25, "domain": 0.25, "plausibility": 0.25}
INT_KEYS = list(WEIGHTS.keys())

def build_prompt(sentence: str) -> str:
    return PROMPT_TEMPLATE.format(sentence=sentence.replace('"', '\\"'))

def parse_json_response(text: str) -> Dict[str, Any]:
    # Strip code fences
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    # Extract the outer JSON object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group(0)
    data = json.loads(text)

    for k in INT_KEYS:
        v = int(data.get(k, 0))
        v = max(1, min(10, v))  # clamp to 1–10
        data[k] = v

    if "total_weighted" not in data:
        tw = sum(data[k] * WEIGHTS[k] for k in INT_KEYS)
        data["total_weighted"] = round(float(tw), 1)

    data["notes"] = str(data.get("notes", "")).strip()
    return data

def score_sentence(llm: GPT, sentence: str, retries: int = 3, sleep_sec: float = 0.2) -> Tuple[Dict[str, Any], float, str, bool]:
    prompt = build_prompt(sentence)
    last_raw = ""
    for attempt in range(1, retries + 1):
        t0 = time.time()
        out = llm.query(prompt)
        dt = time.time() - t0
        last_raw = out
        try:
            parsed = parse_json_response(out)
            return parsed, dt, out, False  # False = not a parse failure
        except Exception as e:
            if attempt == retries:
                return {
                    "fluency": 0, "clarity": 0, "domain": 0,
                    "plausibility": 0,
                    "total_weighted": 0.0, "notes": f"parse_error: {e}"
                }, dt, last_raw, True
            time.sleep(sleep_sec)
    return {
        "fluency": 0, "clarity": 0, "domain": 0,
        "plausibility": 0,
        "total_weighted": 0.0, "notes": "unknown_error"
    }, 0.0, last_raw, True

def _fmt_time(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

def _progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0: return "[" + " " * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def evaluate_csv(in_csv: Path, out_csv: Path, llm: GPT, method_label: str, resume: bool = True, print_every: int =20):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done_idx = set()
    if resume and out_csv.exists():
        with out_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    done_idx.add(int(row["idx"]))
                except Exception:
                    continue

    tasks: List[dict] = []
    with in_csv.open("r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            try:
                idx = int(row["idx"])
            except Exception:
                continue
            if resume and idx in done_idx:
                continue
            sentence = (row.get("sentence") or "").strip()
            if not sentence:
                continue
            tasks.append(row)

    total = len(tasks)
    print(f"[{method_label}] to score: {total} (skipped {len(done_idx)} already in {out_csv.name})")

    with out_csv.open("a", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "idx", "block_idx", "meta", "method", "sentence",
                "fluency", "clarity", "domain", "plausibility",
                "total_weighted", "latency_s", "raw"
            ]
        )
        if f_out.tell() == 0:
            writer.writeheader()

        start = time.time()
        lat_sum = 0.0
        failures = 0

        for i, row in enumerate(tasks, 1):
            idx = int(row["idx"])
            sentence = row["sentence"].strip()

            scores, latency, raw, parse_failed = score_sentence(llm, sentence)
            lat_sum += latency
            if parse_failed:
                failures += 1

            out_row = {
                "idx": idx,
                "block_idx": row.get("block_idx", ""),
                "meta": row.get("meta", ""),
                "method": method_label,
                "sentence": sentence,
                "fluency": scores["fluency"],
                "clarity": scores["clarity"],
                "domain": scores["domain"],
                "plausibility": scores["plausibility"],
                "total_weighted": scores["total_weighted"],
                "latency_s": f"{latency:.2f}",
                "raw": raw.replace("\n", "\\n")[:4000],
            }
            writer.writerow(out_row)

            if (i % print_every == 0) or (i == total):
                avg_lat = lat_sum / i if i else 0.0
                elapsed = time.time() - start
                eta = avg_lat * (total - i)
                bar = _progress_bar(i, total)
                pct = (i / total * 100.0) if total else 100.0
                print(f"{bar} [{method_label}] {i}/{total} ({pct:5.1f}%)  "
                      f"avg_latency={avg_lat:.2f}s  elapsed={_fmt_time(elapsed)}  "
                      f"ETA={_fmt_time(eta)}  failures={failures}",
                      flush=True)
            time.sleep(0.05)

# === (Additional) result summarization utilities ===
SCORE_COLS = ["fluency", "clarity", "domain", "plausibility", "total_weighted"]

def _to_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def summarize_from_csv(csv_paths: List[Path],
                       exclude_zero_total: bool = True,
                       save_path: Path | None = None) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, List[float]]] = {}
    counts_total: Dict[str, int] = {}
    counts_used: Dict[str, int] = {}

    for p in csv_paths:
        if not p.exists():
            print(f"[warn] file not found: {p}")
            continue
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = (row.get("method") or "").strip() or "unknown"
                counts_total[method] = counts_total.get(method, 0) + 1

                tw = _to_float(row.get("total_weighted", "0"))
                if exclude_zero_total and tw == 0.0:
                    continue

                if method not in buckets:
                    buckets[method] = {c: [] for c in SCORE_COLS}
                counts_used[method] = counts_used.get(method, 0) + 1

                for c in SCORE_COLS:
                    buckets[method][c].append(_to_float(row.get(c, "0")))

    result: Dict[str, Dict[str, float]] = {}
    for method, cols in buckets.items():
        result[method] = {}
        for c, arr in cols.items():
            result[method][c] = sum(arr) / len(arr) if arr else float("nan")
        result[method]["N"] = float(counts_used.get(method, 0))
        result[method]["Excluded"] = float(counts_total.get(method, 0) - counts_used.get(method, 0))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["method", "N", "Excluded"] + SCORE_COLS
            writer.writerow(header)
            for method, stats in result.items():
                row = [method, int(stats["N"]), int(stats["Excluded"])] + [f"{stats[c]:.3f}" for c in SCORE_COLS]
                writer.writerow(row)

    return result

def print_summary_table(stats: Dict[str, Dict[str, float]]):
    methods = sorted(stats.keys())
    cols = ["N", "Excluded"] + SCORE_COLS
    col_name = {
        "fluency": "Fluency",
        "clarity": "Clarity",
        "domain": "Domain",
        "plausibility": "Plausibility",
        "total_weighted": "Total",
        "N": "N",
        "Excluded": "Excl."
    }

    head = ["method"] + [col_name[c] for c in cols]
    print("\n=== Score Summary (means) ===")
    print(" | ".join(f"{h:>12}" for h in head))
    print("-" * (14 * len(head)))

    for m in methods:
        row = stats[m]
        vals = [
            f"{int(row['N']):>12}",
            f"{int(row['Excluded']):>12}",
            f"{row['fluency']:.3f}",
            f"{row['clarity']:.3f}",
            f"{row['domain']:.3f}",
            f"{row['plausibility']:.3f}",
            f"{row['total_weighted']:.3f}",
        ]
        print(" | ".join([f"{m:>12}"] + vals))

    if "grow" in stats and "ragwm" in stats:
        diff = stats["grow"]["total_weighted"] - stats["ragwm"]["total_weighted"]
        print(f"\nΔ Total (grow - ragwm): {diff:.3f}")


if __name__ == "__main__":
    CFG_PATH = Path("./model_configs/gpt4_config.json")
    cfg = load_json(CFG_PATH)
    llm = GPT(cfg)

    DATASET = ["trec-covid", "nfcorpus", "nq", "hotpotqa", "msmarco"]
    for dataset in DATASET:
        print(f"== DATASET: {dataset} ==")
        base = Path(f"output/wm_generate/{dataset}/10")
        in_grow  = base / "wminject_sentences_grow.csv"
        in_ragwm = base / "wminject_sentences_ragwm.csv"

        out_grow  = base / "wm_scores_grow.csv"
        out_ragwm = base / "wm_scores_ragwm.csv"

        print("== scoring grow ==")
        evaluate_csv(in_grow, out_grow, llm, method_label="grow", resume=True, print_every=20)

        # print("== scoring ragwm ==")
        # evaluate_csv(in_ragwm, out_ragwm, llm, method_label="ragwm", resume=True, print_every=20)

        summary_csv = base / "wm_scores_summary.csv"
        stats = summarize_from_csv([out_grow, out_ragwm], exclude_zero_total=True, save_path=summary_csv)
        print_summary_table(stats)
        print(f"\nSaved summary to: {summary_csv}")

        print("Done.")
