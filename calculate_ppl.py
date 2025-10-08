# ppl_collect.py
# -*- coding: utf-8 -*-
import os, csv, json, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== (Ref) LLaMA loader & PPL calculator ======
def load_llama(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
               gpu_id: int = 0,
               load_in_8bit: bool = False):
    torch.cuda.set_device(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"device_map": {"": gpu_id}}
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def perplexity(model, tokenizer, text: str, stride: int = 512, max_length: int = 2048) -> float:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)

    nlls, total_tokens = [], 0
    if input_ids.shape[1] <= max_length:
        out = model(input_ids, labels=input_ids)
        nlls.append(out.loss.float() * input_ids.shape[1])
        total_tokens += input_ids.shape[1]
    else:
        L = input_ids.shape[1]
        for s in range(0, L, stride):
            e = min(s + max_length, L)
            chunk = input_ids[:, s:e]
            out = model(chunk, labels=chunk)
            nlls.append(out.loss.float() * (e - s))
            total_tokens += (e - s)
            if e == L: break
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()

# ====== Read only the `sentence` column from CSV ======
def read_sentences(csv_path: Path, sentence_col: str = "sentence", limit: int | None = None):
    sents = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Auto-detect when the sentence column name differs
        if sentence_col not in reader.fieldnames:
            for cand in ("sentence", "text", "sent"):
                if cand in reader.fieldnames:
                    sentence_col = cand
                    break
        for row in reader:
            s = (row.get(sentence_col) or "").strip()
            if s:
                sents.append(s)
                if limit and len(sents) >= limit:
                    break
    return sents

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True,
                    choices=["trec-covid","nfcorpus","nq","hotpotqa","msmarco"])
    ap.add_argument("--base", type=str, default="knot-main/output/wm_generate")
    ap.add_argument("--subdir", type=str, default="10")  # wminject folder level
    ap.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--limit_clean", type=int, default=None, help="If None, take all (usually ~2000)")
    ap.add_argument("--limit_wm", type=int, default=None, help="If None, take all (usually ~50)")
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    ds = args.dataset
    root = Path(args.base) / ds / args.subdir

    # File paths
    clean_csv = root / "clean_sentences.csv"
    grow_csv  = root / "wminject_sentences_grow.csv"  # filename kept as-is (data label), not a project path
    # Note: some files may be misspelled as 'setences'; check both
    ragwm_csv = root / "wminject_setences_ragwm.csv"
    if not ragwm_csv.exists():
        ragwm_csv = root / "wminject_sentences_ragwm.csv"

    assert clean_csv.exists(), f"Missing: {clean_csv}"
    assert grow_csv.exists(),  f"Missing: {grow_csv}"
    assert ragwm_csv.exists(), f"Missing: {ragwm_csv}"

    # Load model
    model, tokenizer = load_llama(args.model_name, args.gpu_id, args.load_in_8bit)

    # Read sentences
    clean_sents = read_sentences(clean_csv, limit=args.limit_clean)
    grow_sents  = read_sentences(grow_csv,  limit=args.limit_wm)
    ragwm_sents = read_sentences(ragwm_csv, limit=args.limit_wm)

    print(f"[{ds}] #clean={len(clean_sents)}, #grow={len(grow_sents)}, #ragwm={len(ragwm_sents)}")

    # Compute PPL
    clean_ppl, grow_ppl, ragwm_ppl = [], [], []
    for i, s in enumerate(clean_sents, 1):
        clean_ppl.append(perplexity(model, tokenizer, s, args.stride, args.max_length))
        if i % 100 == 0: print(f" clean {i}/{len(clean_sents)}")

    for i, s in enumerate(grow_sents, 1):
        grow_ppl.append(perplexity(model, tokenizer, s, args.stride, args.max_length))
        if i % 25 == 0: print(f" grow {i}/{len(grow_sents)}")

    for i, s in enumerate(ragwm_sents, 1):
        ragwm_ppl.append(perplexity(model, tokenizer, s, args.stride, args.max_length))
        if i % 25 == 0: print(f" ragwm {i}/{len(ragwm_sents)}")

    print(f"done. examples → clean[0]={clean_ppl[0]:.3f} | grow[0]={grow_ppl[0]:.3f} | ragwm[0]={ragwm_ppl[0]:.3f}")

    # Save
    out_path = Path(args.out) if args.out else root / "ppl_values.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": ds,
            "model": args.model_name,
            "clean": clean_ppl,
            "grow": grow_ppl,
            "ragwm": ragwm_ppl
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved PPL lists → {out_path}")

if __name__ == "__main__":
    main()


"""
python calculate_ppl.py \
  --dataset trec-covid \
  --base /mnt/ssd/TSF/knot-main/output/wm_generate \
  --subdir 10 \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --gpu_id 0

python calculate_ppl.py \
  --dataset nfcorpus \
  --base /mnt/ssd/TSF/knot-main/output/wm_generate \
  --subdir 10 \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --gpu_id 0

python calculate_ppl.py \
  --dataset nq \
  --base /mnt/ssd/TSF/knot-main/output/wm_generate \
  --subdir 10 \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --gpu_id 0

python calculate_ppl.py \
  --dataset hotpotqa \
  --base /mnt/ssd/TSF/knot-main/output/wm_generate \
  --subdir 10 \
  --model_name meta-llama-3.1-8b-instruct \
  --gpu_id 0  

python calculate_ppl.py \
  --dataset msmarco \
  --base /mnt/ssd/TSF/knot-main/output/wm_generate \
  --subdir 10 \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --gpu_id 0
"""
