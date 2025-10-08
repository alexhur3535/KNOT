# count_er.py
import os
import json
from pathlib import Path

BASE_DIR = Path("./output/wm_prepare")  # Change to absolute path if needed

def safe_read_json(path: Path):
    """Safely read a JSON file and handle missing or invalid files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON decode error in {path}: {e}")
        return None

def count_entities_relations(dataset_dir: Path):
    """Count the number of unique entities and relations in a given dataset directory."""
    ent_path = dataset_dir / "entities_dict_llm.json"
    rel_path = dataset_dir / "relation_list_llm.json"

    entities = safe_read_json(ent_path)
    relations = safe_read_json(rel_path)

    # --- entities_dict_llm.json: expected format = dict (entity_name -> type) ---
    if isinstance(entities, dict):
        n_entities = len(entities)
    elif isinstance(entities, list):
        # In rare cases, it might be a list; collect unique entity names
        # (Normally, it should be a dict)
        names = []
        for x in entities:
            if isinstance(x, dict):
                names.extend(list(x.keys()))
        n_entities = len(set(names)) if names else len(entities)
    else:
        n_entities = 0

    # --- relation_list_llm.json: expected format = list of [E1, E2, R] triples ---
    if isinstance(relations, list):
        n_relations = len(relations)
    else:
        n_relations = 0

    return n_entities, n_relations

def main():
    """Main routine for counting entities and relations across all datasets."""
    if not BASE_DIR.exists():
        print(f"[ERROR] Base directory not found: {BASE_DIR.resolve()}")
        return

    dataset_dirs = sorted(
        [d for d in BASE_DIR.iterdir() if d.is_dir()]
    )
    if not dataset_dirs:
        print(f"[WARN] No dataset folders found under {BASE_DIR.resolve()}")
        return

    print("✅ Entity / Relation Count per Dataset\n")
    print(f"{'Dataset':<20} {'#Entities':>10} {'#Relations':>12}")
    print("-" * 44)

    total_e, total_r = 0, 0
    for d in dataset_dirs:
        ne, nr = count_entities_relations(d)
        total_e += ne
        total_r += nr
        print(f"{d.name:<20} {ne:>10} {nr:>12}")

    print("-" * 44)
    print(f"{'TOTAL':<20} {total_e:>10} {total_r:>12}")

if __name__ == "__main__":
    main()
