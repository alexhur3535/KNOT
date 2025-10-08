# attack_and_verify.py
import sys
sys.path.append('/mnt/ssd/TSF/knot-main')  # adjust if needed

import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from copy import deepcopy

from src.models import GPT
from src.utils import load_json, save_json

# === Attack prompts ===
PARAPHRASE_PROMPT = 'Paraphrase the following sentences:\n\n{TEXT}'
UNRELATED_PROMPT = '''You are a helpful assistant. Below is a text which may contain unrelated or incoherent sentences.
Your task is to return the text **after** removing any unrelated or incoherent parts.
If every sentence is relevant and coherent, return the **original text exactly as it is** — do not add explanations or comments.

Text:
{TEXT}'''
# Summary attack prompt: encourage short, generic phrasing → may omit/soften E1/E2
SUMMARIZE_PROMPT = '''Summarize the following text into a single concise sentence (<= 25 words).
Use generic references where possible (e.g., "the official", "the project") instead of repeating exact names.
Do not add new facts or quotes.

Text: {TEXT}'''

# === Watermark detection prompt ===
WATERMARK_CHECK = '''
You are an expert in knowledge graphs. Your task is to decide if the text truly expresses the target relation R1 between E1 and E2.

Evaluation rules:
- Both E1 and E2 must appear in the text (exact strings or clear references).
- Focus on the semantic meaning of R1:
  * Accept "yes" only if the text uses the same phrase as R1, or a clear synonym/paraphrase with the SAME meaning.
  * Reject if the connection is expressed with generic or weak expressions (e.g., "related to", "associated with", "suggested for", "used for") that do not specifically convey the intent of R1.
- Directionality must match (E1 →R1→ E2).
- If the meaning is different, vague, hedged, or only loosely connected, answer "no".

Input:
- rag_doc: ({rag_doc})
- R1: ({R1})
- E1: ({E1})
- E2: ({E2})

Output:
Reply strictly with lowercase "yes" or "no".
'''

# ===== Utils =====
def _attack_prompt(mode: str, text: str) -> str:
    if mode == "paraphrase":
        return PARAPHRASE_PROMPT.format(TEXT=text)
    elif mode == "unrelated":
        return UNRELATED_PROMPT.format(TEXT=text)
    elif mode == "summary":
        return SUMMARIZE_PROMPT.format(TEXT=text)
    else:
        raise ValueError(f"Unknown attack mode: {mode}")

def run_attack(llm: GPT, text: str, mode: str) -> str:
    prompt = _attack_prompt(mode, text)
    out = llm.query(prompt)
    return (out or "").strip()

def run_attack_sequence(llm: GPT, text: str, modes: List[str]) -> Tuple[str, List[str]]:
    """Apply attack modes sequentially in order; also return each intermediate output."""
    intermediates: List[str] = []
    cur = text
    for m in modes:
        cur = run_attack(llm, cur, m)
        intermediates.append(cur)
        time.sleep(0.12)  # mitigate rate limits
    return cur, intermediates

def watermark_check(llm: GPT, rag_doc: str, R: str, E1: str, E2: str) -> str:
    prompt = WATERMARK_CHECK.format(rag_doc=rag_doc, R1=R, E1=E1, E2=E2)
    out = llm.query(prompt) or ""
    out = re.sub(r"[^a-zA-Z]", "", out).lower()
    return "yes" if out.startswith("yes") else "no"

def count_yes(items: List[Dict[str, Any]], key: str) -> int:
    yes = 0
    for x in items:
        v = (x.get(key, "") or "").strip().lower()
        if v in ("yes", "yes.", "y", "true", "1"):
            yes += 1
    return yes

def load_baseline(input_json: Path, limit: int | None) -> tuple[int, int, List[Dict[str, Any]]]:
    data = load_json(input_json)
    if limit is not None:
        data = data[:limit]
    n_yes = count_yes(data, "result")
    return n_yes, len(data), data

def attack_and_verify_once(input_json: Path, output_json: Path, llm: GPT, attack_mode: str, limit: int | None) -> tuple[int, int]:
    """Apply a single attack mode (paraphrase/unrelated/summary)."""
    data = load_json(input_json)
    if limit is not None:
        data = data[:limit]

    out_list: List[Dict[str, Any]] = []
    for i, item in enumerate(data, 1):
        rag_answer = item.get("rag_answer", "")
        tup = item.get("tuple", {})
        E1, E2, R = tup.get("E1", ""), tup.get("E2", ""), tup.get("R", "")

        attacked_answer = run_attack(llm, rag_answer, attack_mode)
        attacked_result = watermark_check(llm, attacked_answer, R, E1, E2)

        new_item = deepcopy(item)
        new_item["attack_mode"] = attack_mode
        new_item["attacked_answer"] = attacked_answer
        new_item["attacked_result"] = attacked_result
        new_item["attacked_flag"] = 1 if attacked_result == "yes" else 0
        out_list.append(new_item)

        if i % 10 == 0 or i == len(data):
            print(f"  - {attack_mode}: {i}/{len(data)}")

        time.sleep(0.12)

    save_json(out_list, output_json)
    yes_cnt = count_yes(out_list, "attacked_result")
    return yes_cnt, len(out_list)

def attack_and_verify_all_chain(input_json: Path, output_json: Path, llm: GPT, limit: int | None) -> tuple[int, int]:
    """
    Apply all three attacks in sequence: paraphrase → unrelated → summary.
    Also store intermediate outputs as attacked_answer_step1/2/3.
    """
    chain = ["paraphrase", "unrelated", "summary"]
    data = load_json(input_json)
    if limit is not None:
        data = data[:limit]

    out_list: List[Dict[str, Any]] = []
    for i, item in enumerate(data, 1):
        rag_answer = item.get("rag_answer", "")
        tup = item.get("tuple", {})
        E1, E2, R = tup.get("E1", ""), tup.get("E2", ""), tup.get("R", "")

        final_answer, mids = run_attack_sequence(llm, rag_answer, chain)
        attacked_result = watermark_check(llm, final_answer, R, E1, E2)

        new_item = deepcopy(item)
        new_item["attack_mode"] = "all"
        # Record per-step outputs
        if len(mids) >= 1: new_item["attacked_answer_step1_paraphrase"] = mids[0]
        if len(mids) >= 2: new_item["attacked_answer_step2_unrelated"] = mids[1]
        if len(mids) >= 3: new_item["attacked_answer_step3_summary"] = mids[2]
        new_item["attacked_answer"] = final_answer
        new_item["attacked_result"] = attacked_result
        new_item["attacked_flag"] = 1 if attacked_result == "yes" else 0
        out_list.append(new_item)

        if i % 10 == 0 or i == len(data):
            print(f"  - all(chain): {i}/{len(data)}")

        time.sleep(0.12)

    save_json(out_list, output_json)
    yes_cnt = count_yes(out_list, "attacked_result")
    return yes_cnt, len(out_list)

def run_for_datasets(
    datasets: List[str],
    base_dir: str = "output/wm_generate",
    model_tag: str = "gpt3.5",
    topk_dir: str = "10",
    limit_each: int | None = 30,
    cfg_path: str = "./model_configs/gpt3.5_config.json",
):
    cfg = load_json(Path(cfg_path))
    llm = GPT(cfg)

    overall = {
        "baseline_yes": 0,
        "paraphrase_yes": 0,
        "unrelated_yes": 0,
        "summary_yes": 0,
        "all_yes": 0,
        "total": 0
    }

    print("=== Multi-Dataset Attack & Verify ===")
    for ds in datasets:
        ds_dir = Path(base_dir) / ds / model_tag / topk_dir
        in_path = ds_dir / "verify_answers.json"
        if not in_path.exists():
            print(f"[WARN] missing: {in_path}")
            continue

        print(f"\n== DATASET: {ds} ==")
        # 0) Baseline
        base_yes, base_total, _ = load_baseline(in_path, limit_each)
        print(f"[Baseline]              {base_yes}/{base_total}")

        # 1) Paraphrase
        para_out = ds_dir / "verify_answers_paraphrase.json"
        para_yes, para_total = attack_and_verify_once(in_path, para_out, llm, "paraphrase", limit_each)
        print(f"[Paraphrase attack]     {para_yes}/{para_total}")

        # 2) Unrelated-removal
        unrel_out = ds_dir / "verify_answers_unrelated.json"
        unrel_yes, unrel_total = attack_and_verify_once(in_path, unrel_out, llm, "unrelated", limit_each)
        print(f"[Unrelated removal]     {unrel_yes}/{unrel_total}")

        # 3) Summarization
        sum_out = ds_dir / "verify_answers_summary.json"
        sum_yes, sum_total = attack_and_verify_once(in_path, sum_out, llm, "summary", limit_each)
        print(f"[Summarization]         {sum_yes}/{sum_total}")

        # 4) All-chain: paraphrase → unrelated → summary
        all_out = ds_dir / "verify_answers_all.json"
        all_yes, all_total = attack_and_verify_all_chain(in_path, all_out, llm, limit_each)
        print(f"[All (chain 3-step)]    {all_yes}/{all_total}")

        # Aggregate
        overall["baseline_yes"]   += base_yes
        # overall["paraphrase_yes"] += para_yes
        # overall["unrelated_yes"]  += unrel_yes
        # overall["summary_yes"]    += sum_yes
        overall["all_yes"]        += all_yes
        overall["total"]          += base_total  # use the same N

    print("\n=== Overall Summary (sum across datasets) ===")
    print(f"Baseline total:          {overall['baseline_yes']}/{overall['total']}")
    print(f"Paraphrase total:        {overall['paraphrase_yes']}/{overall['total']}")
    print(f"Unrelated-removal total: {overall['unrelated_yes']}/{overall['total']}")
    print(f"Summarization total:     {overall['summary_yes']}/{overall['total']}")
    print(f"All (chain) total:       {overall['all_yes']}/{overall['total']}")

if __name__ == "__main__":
    # Add as needed: "trec-covid", "nq", "nfcorpus", etc.
    DATASETS = ["trec-covid", "nq", "nfcorpus", "hotpotqa", "msmarco"]
    run_for_datasets(
        datasets=DATASETS,
        # Path rule: output/wm_generate/<dataset>/gpt3.5/10/verify_answers.json
        base_dir="output/wm_generate",
        model_tag="gpt3.5",
        topk_dir="10",
        limit_each=30,  # if file has fewer than 30 entries, uses its length automatically
        cfg_path="./model_configs/gpt3.5_config.json",
    )
