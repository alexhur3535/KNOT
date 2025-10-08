# -*- coding: utf-8 -*-
"""
knot_llm_relation_gen.py
- 각 item에서 E1, E3, mediator의 R12/R23를 불러와
- LLM에게 "R12, R23과 겹치지 않으면서 plausible한 새로운 relation"을 생성 요청
- 결과를 [E1, E2, R] 형식으로 저장
"""

import argparse, json, sys
from pathlib import Path
from src.models import create_model

def make_prompt(e1: str, e3: str, r12: list, r23: list) -> str:
    forbid = ", ".join(r12 + r23) if (r12 or r23) else "(none)"
    return f"""You are a knowledge graph relation generator.

Task:
Create one new, unique relation phrase that plausibly connects E1 to E2 in natural English text.
Constraints:
- Do NOT repeat or paraphrase any of the following relations: {forbid}
- The relation must explicitly and directionally link E1 → E2.
- Keep it short (1–3 words), specific, and plausible.
- Avoid generic terms like 'RELATED_TO', 'ASSOCIATED_WITH', 'CONNECTED_TO'.

Entities:
- E1: {e1}
- E2: {e3}

Output format:
- R: <your phrase>
"""

def load_items(dataset_id: str):
    base = Path("output") / "wm_prepare" / dataset_id
    in_links = base / "wm_links_v3.json"
    if not in_links.exists():
        raise FileNotFoundError(f"not found: {in_links}")
    with in_links.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", required=True)
    ap.add_argument("--max_blocks", type=int, default=50)
    args = ap.parse_args()

    # ---- NOTE: llm 객체는 미리 정의되어 있어야 함 ----
    model_config_path = f'model_configs/gpt3.5_config.json'
    llm = create_model(model_config_path)
    try:
        llm  # noqa: F821
    except NameError:
        print("[ERROR] `llm` object is not defined in this runtime. Please provide an LLM client with llm.query(prompt).", file=sys.stderr)
        sys.exit(1)

    items = load_items(args.dataset_id)
    n = min(args.max_blocks, len(items))

    results = []
    for idx in range(n):
        it = items[idx]
        e1, e3 = it.get("E1", ""), it.get("E3", "")
        mediators = it.get("mediators", [])
        r12, r23 = [], []
        for m in mediators or []:
            r12.extend(m.get("R12", []))
            r23.extend(m.get("R23", []))

        prompt = make_prompt(e1, e3, r12, r23)
        raw = llm.query(prompt)
        relation = raw.strip()

        # --- 후처리: "R:" 접두어 제거 ---
        if relation.lower().startswith("r: "):
            relation = relation[3:].strip()

        # 화면 출력
        print(f"block {idx+1}")
        print(f"E1: {e1}")
        print(f"E2: {e3}")
        print(f"R: {relation}")
        print()

        # 결과 저장용 (E1, E2, R 배열 형태)
        results.append([e1, e3, relation])

    # 저장 경로
    out_file = Path("output") / "wm_prepare" / args.dataset_id / "wmunit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved {len(results)} watermark tuples -> {out_file}")

if __name__ == "__main__":
    main()

# python find_wm_relation.py --dataset_id trec-covid --max_blocks 50
# python find_wm_relation.py --dataset_id nfcorpus --max_blocks 50
# python find_wm_relation.py --dataset_id nq --max_blocks 50
# python find_wm_relation.py --dataset_id hotpotqa --max_blocks 50
# python find_wm_relation.py --dataset_id msmarco --max_blocks 50