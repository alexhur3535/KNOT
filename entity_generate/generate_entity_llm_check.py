import json
import os
import time
import pickle
import traceback
from datasets import load_dataset

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

import sys
import argparse
import random
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print(project_root)
sys.path.append(project_root)
from src.utils import load_beir_datasets, load_json, save_json

from src.utils import Log, file_exist


def data_prepare(dataname, dataset):
    """Normalize datasets into a {doc_id: text} mapping used by the pipeline."""
    ndataset = {}
    if dataname in ['nq', 'msmarco', 'hotpotqa', 'nfcorpus', 'trec-covid']:
        for key, value in dataset.items():
            ndataset[key] = value['text']
    elif dataname == 'closed_qa':
        for i, item in enumerate(dataset):
            ndataset[str(i)] = f"{item['instruction']}. {item['context']}"
    return ndataset


def generate_entity(dataset, basepath, prob=0.01):
    """Sample documents with probability `prob`, extract graph nodes/relations with an LLM, and checkpoint results."""
    entities_dict = {}
    node_type_dict = {}
    relation_type_dict = {}
    unique_relation = []
    current_key = ''
    tcount = 0

    # Try to load previously saved intermediate results (resume support)
    try:
        entities_dict, node_type_dict, relation_type_dict, unique_relation, current_key = load_checkpoint(basepath)
        print(current_key, type(current_key))
        print("Successfully loaded saved checkpoint.")
    except Exception as e:
        print(f"Error occurred while loading checkpoint: {e}")
        traceback.print_exc()
    print(current_key, type(current_key))

    # Iterate over keys in sorted order to enable deterministic/resumable runs
    for i, key in enumerate(sorted(dataset.keys())):
        # logger.info(f'i: {i}, key:{key}')
        if current_key != '' and key < current_key:
            # Skip keys that were already processed
            # print(f'key already processed, key:{key}, current_key:{current_key}')
            continue

        # Subsample the dataset by probability
        if random.random() > prob:
            # print(f'skipped by prob: {prob}')
            continue

        item = dataset[key]
        print(f"Processing item {i}: key = {key}, tcount: {tcount} ")
        # print(f'item: {item}')
        tcount += 1

        # Convert single document to a graph document
        documents = [Document(page_content=item)]
        print(f'documents:{documents}')
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        nodes_list = graph_documents[0].nodes

        # Process nodes
        # print(f'node_list:{nodes_list}')
        for node in nodes_list:
            node_type_dict[node.type] = node_type_dict.get(node.type, 0) + 1
            if node.id not in entities_dict:
                entities_dict[node.id] = node.type

        relations_list = graph_documents[0].relationships

        # Process relations
        for relation in relations_list:
            relation_type_dict[relation.type] = relation_type_dict.get(relation.type, 0) + 1
            triplet = [relation.source.id, relation.target.id, relation.type]
            if triplet not in unique_relation:
                unique_relation.append(triplet)

        # Periodically save intermediate results (lightweight checkpointing)
        if tcount % 20 == 1:
            print(i, time.time())
            save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation, key, basepath)

    # Final checkpoint after finishing the loop
    save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation, key, basepath)
    print("All documents have been processed.")
    return True


def save_checkpoint(entities_dict, node_type_dict, relation_type_dict, unique_relation, current_key, basepath):
    """Persist current progress and aggregates to disk."""
    entities_dict_path = os.path.join(basepath, 'entities_dict_llm.json')
    entity_type_path = os.path.join(basepath, 'entity_type_llm.json')
    relation_type_path = os.path.join(basepath, 'relation_type_llm.json')
    relation_list_path = os.path.join(basepath, 'relation_list_llm.json')
    checkpoint_file = os.path.join(basepath, 'checkpoint.json')

    save_json(current_key, checkpoint_file)
    save_json(entities_dict, entities_dict_path)
    save_json(node_type_dict, entity_type_path)
    save_json(relation_type_dict, relation_type_path)
    save_json(unique_relation, relation_list_path)


def load_checkpoint(basepath):
    """Load previously saved progress and aggregates."""
    entities_dict_path = os.path.join(basepath, 'entities_dict_llm.json')
    entity_type_path = os.path.join(basepath, 'entity_type_llm.json')
    relation_type_path = os.path.join(basepath, 'relation_type_llm.json')
    relation_list_path = os.path.join(basepath, 'relation_list_llm.json')
    checkpoint_file = os.path.join(basepath, 'checkpoint.json')

    entities_dict = load_json(entities_dict_path)
    node_type_dict = load_json(entity_type_path)
    relation_type_dict = load_json(relation_type_path)
    unique_relation = load_json(relation_list_path)
    current_key = load_json(checkpoint_file)

    return entities_dict, node_type_dict, relation_type_dict, unique_relation, current_key


def parse_args():
    parser = argparse.ArgumentParser(description='Generate entities and relations from dataset')
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='gpt3.5')
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever", "contriever-msmarco", "ance"])
    parser.add_argument('--eval_dataset', type=str, default='hotpotqa', help='BEIR dataset to evaluate',
                        choices=['trec-covid', 'nfcorpus', 'nq', 'msmarco', 'hotpotqa'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--basepath', type=str, default='/mnt/ssd/TSF/knot-main/output/wm_prepare')
    parser.add_argument('--dataset_prob', type=float, default=0.01, help='Sampling probability per document (e.g., 1.0 for full pass on NFCorpus)')
    return parser.parse_args()


if __name__ == '__main__':
    print("generate_entity")
    args = parse_args()

    # LLM setup
    if args.model_config_path is None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
    config = load_json(args.model_config_path)

    api_keys = config["api_key_info"]["api_keys"][0]
    api_base = config["api_key_info"]["api_base"][0]
    os.environ["OPENAI_API_KEY"] = api_keys
    os.environ["OPENAI_API_BASE"] = api_base

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm_transformer = LLMGraphTransformer(llm=llm)

    # Load dataset and normalize format
    if args.eval_dataset == 'closed_qa':
        train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        closed_qa_dataset = train_dataset.filter(lambda example: example['category'] == 'closed_qa')
        ndataset = data_prepare('closed_qa', closed_qa_dataset)
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)
        ndataset = data_prepare(args.eval_dataset, corpus)

    # Logging setup
    basepath = os.path.join(args.basepath, args.eval_dataset)
    LOG_FILE = os.path.join(basepath, 'log.log')
    file_exist(LOG_FILE)

    logger = Log(log_file=LOG_FILE).get(__file__)
    logger.info(f'args:{args}')
    print(LOG_FILE)

    # Timing
    start_time = time.time()
    generate_entity(ndataset, basepath, prob=args.dataset_prob)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"✅ Entity generation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


# NFCorpus: 3,633 items → use prob=1.0 for full pass
#   python entity_generate/generate_entity_llm_check.py --eval_dataset 'nfcorpus' --dataset_prob 1
# TREC-COVID: 171,332 items → use prob=0.03
#   python entity_generate/generate_entity_llm_check.py --eval_dataset 'trec-covid' --dataset_prob 0.03
# NQ: 2,681,468 items → use prob=0.0019
#   python entity_generate/generate_entity_llm_check.py --eval_dataset 'nq' --dataset_prob 0.0019
# MS MARCO: 8,841,823 items → use prob=0.00057
#   python entity
