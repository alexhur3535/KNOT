import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.append('/mnt/ssd/TSF/knot-main')  # 필요 시 조정

def export_from_wmunit_inject(
    in_json: str | Path,
    out_csv: str | Path,
    meta_value: str = "knot",
    dedup: bool = False,          # (block_idx, sentence) 기준 중복 제거
) -> int:
    """
    wmuint_inject.json -> wminject_sentences_*.csv
    CSV 헤더: idx, block_idx, meta, sentence
    반환: 저장된 문장 수
    """
    in_json = Path(in_json)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with in_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    seen = set()
    global_idx = 0
    # print(f"len(data): {len(data)}")

    # 각 block: [ [head, tail, rel], [ids...], [ [sentence, 1, 1], ... ] ]
    for block_idx, block in enumerate(data):
        if not (isinstance(block, list) and len(block) >= 3):
            continue
        texts = block[2][0] if isinstance(block[2][0], list) else []
        # print(f"type(texts): {type(texts)}")
        # print(block)
        # print(block[2])
        # print(texts)
        # quit()
        for item in texts:
            # item이 ["문장", 1, 1] 형식일 때 첫 원소 사용
            if isinstance(item, list) and item and isinstance(item[0], str):
                sent = item[0].strip()
            elif isinstance(item, str):
                sent = item.strip()
            else:
                continue
            if not sent:
                continue

            if dedup:
                key = (block_idx, sent)
                if key in seen:
                    continue
                seen.add(key)

            rows.append({
                "idx": global_idx,
                "block_idx": block_idx,
                "meta": meta_value,
                "sentence": sent
            })
            global_idx += 1

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "block_idx", "meta", "sentence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} sentences → {out_csv}")
    return len(rows)


if __name__ == "__main__":
    # DATASET = ["trec-covid", "nfcorpus", "nq", "hotpotqa", "msmarco"]
    DATASET = ["nfcorpus", "nq", "hotpotqa", "msmarco"]
    for dataset in DATASET:
        base = Path(f"output/wm_generate/{dataset}/10")
        src = base / "wmuint_inject.json"
        dst = base / "wminject_sentences_grow.csv"
        export_from_wmunit_inject(src, dst, meta_value="grow", dedup=False)
