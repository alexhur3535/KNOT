import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add project root directory to sys.path
sys.path.append(project_root)
from pathlib import Path
import argparse
 
from watermark_role import Advisor, Visiter, Checker 

from rag.vectorstore import  VectorStore, check_collection
from src.utils import load_json, save_json, find_substrings_containing_sentences,Log,extract_doc_list,extract_doc, file_exist
from src.models import create_model
from src.utils import load_beir_datasets, load_models

import re
import copy 
import torch
import numpy as np
import random

seed = 633

torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Create a fallback log file path so logging works even when imported via `import`
if 'logger' not in globals():
    _fallback_log = os.path.join(os.environ.get("GROW_LOGDIR", "/mnt/ssd/TSF/knot-main/output"),
                                 "main_task", "import_fallback.log")
    Path(os.path.dirname(_fallback_log)).mkdir(parents=True, exist_ok=True)
    logger = Log(log_file=_fallback_log).get(__file__)

def doc_llm_run_mutual(doc_path, mutual_times=0):
    """
    Example:
        python  wm_generate/role.py  --doc 1 --auto 2
    """
    wmunit_list = load_json(watermark_unit_path)
    
    wmunit_doc_list = []
    try:
        wmunit_doc_list = load_json(doc_path)
    except FileNotFoundError:
        logger.error(f'{doc_path} does not exist!')

 
    advisor.K = args.adv_per_wmunit
 

    def get_WT_list(wmunit):
        advisor.wm_unit = wmunit
        len_WT = 0
        WT_list = []
        count = 0
        while len_WT != advisor.K and count < 10:
            WTs = advisor.get_document()
            count+=1
            try:
                if isinstance( WTs, str):
                    logger.info(f'count: {count} ,WTs : {WTs}')
                    WT_list= extract_doc_list(WTs)
                    len_WT = len(WT_list)
                    logger.info(f'count: {count} ,{wmunit} len_WT {len_WT}; {WTs}; {WT_list }')
            except Exception as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # Continue loop after catching exception
            
        return WT_list

    def direct_check(wmunit, WT_list ):
        WT_flag = []
        for WT in WT_list:
            checker.rag_document = WT
            checker.wm_unit = wmunit
            result = checker.check_wm()
    
            if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]:
                WT_flag.append(1)
            elif result in ['No', 'no', 'NO', 'No.','no.','NO.']:
                WT_flag.append(0)
            else:
                WT_flag.append(2)
        
        return WT_flag
    
    def simulate_check(wmunit, WT_list ):
        WT_flag = []
        for WT in WT_list:
            WE = visiter.ask_wm_with_wt(WT)
            checker.rag_document = WE
            checker.wm_unit = wmunit
            result = checker.check_wm()
    
            if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]:
                WT_flag.append(1)
            elif result in ['No', 'no', 'NO', 'No.','no.','NO.']:
                WT_flag.append(0)
            else:
                WT_flag.append(2)
        
        return WT_flag
 
    def get_WT_interation_both_list(wmunit, WT_list, mtimes):
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit
        WT_list_interaction = []

        for WT in WT_list:
            checker.rag_document = WT
            WD1 = checker.check_wm()

            WE = visiter.ask_wm_with_wt(WT)
            checker.rag_document = WE
            WD2 = checker.check_wm()
 
            flag = 0
            true_list =  ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]
            while  WD1 not in true_list or WD2 not in true_list or WT in WT_list_interaction:
                if flag >mtimes : break
                flag += 1
                logger.info(f'mutual_time: {flag}, WD1:{WD1}, WD2:{WD2}, WT:{WT}, WE:{WE}')
                try:
                    WT = advisor.get_document_feedback_both(WT=WT, WE=WE, WD1=WD1, WD2=WD2)
                    logger.info(f'before WT for advisor with feedback: {WT }')
                    WT = extract_doc(WT) 
                    logger.info(f'after WT for advisor with feedback: {WT }')
                    checker.rag_document = WT
                    WD1 = checker.check_wm()

                    WE = visiter.ask_wm_with_wt(WT)
                    checker.rag_document = WE
                    WD2 = checker.check_wm()
 
                except Exception as e:
                    logger.info(f"Error encountered: {e}. Retrying...")
                    continue  # Continue loop after catching exception

            WT_list_interaction.append(WT)
        logger.info(f'WT_list_interaction: {WT_list_interaction}')
        return WT_list_interaction
 
    logger.info(f"len wmunit: {len(wmunit_list)}")
    for item, wmunit in enumerate(wmunit_list):
        if  any(wmunit == wl[0] for wl in wmunit_doc_list): 
            print(item,wmunit)
            continue
        logger.info(f'item:{item},wmunit:{wmunit}')
        wmunit_doc = []
        WT_list = get_WT_list(wmunit)

        if mutual_times != 0 :
            WT_list = get_WT_interation_both_list(wmunit, WT_list, mutual_times)

        WT_direct_flag = direct_check(wmunit, WT_list)
        WT_simulate_flag = simulate_check(wmunit, WT_list)

        wmunit_doc.append(wmunit) 
        wmunit_doc.append(WT_list)
        wmunit_doc.append(WT_direct_flag)
        wmunit_doc.append(WT_simulate_flag)
        # if  wmunit_doc not in wmunit_doc_list:
        wmunit_doc_list.append(wmunit_doc)
        logger.info(f'save path :{doc_path}')
        save_json(wmunit_doc_list, doc_path)
        
    wmdoclen = len(wmunit_doc_list)
    wmlen = len(wmunit_list)
    # logger.info(f'wm len {wmlen}: wm doc len {wmdoclen}')
    # assert(wmlen == wmdoclen)
    
    save_json(wmunit_doc_list, doc_path)

    return  True


def wm_inject_run(doc_path, loc_path, aim_count = 50 ):
    """
    Inputs:
        doc_path: path to the JSON file containing candidate watermark units and WT lists
        db:       (implicit via visiter.vectorstore) target vector DB
    Output:
        loc_path: path to the JSON file with injection locations
    """

    wmunit_doc_list = load_json(doc_path)
    wmunit_inject_loc = []
     
    count_doc = len(wmunit_doc_list)
    
    aim_wmunit_list = []

    def filter_wmunit(wmunit_doc_list, count_doc, aim_count):
        """
        Filter watermark units that have two 'Yes' checks (direct + simulate).
        Falls back to partially passing units if not enough are available.
        """
        awmunit = []
        uawmunit = []
        for i in range(count_doc):
            
            rag_document = []
            rag_document = [j for j in range(len(wmunit_doc_list[i][1])) if wmunit_doc_list[i][2][j] == 1 and wmunit_doc_list[i][3][j] == 1]
            if len( rag_document ) == 0: 
                uawmunit.append(wmunit_doc_list[i])
                continue
            awmunit.append(wmunit_doc_list[i])
            
        aim_len = len(awmunit)
        # Logging quick stats/debug info
        print("total_units:", len(wmunit_doc_list))
        print("sentences_per_unit (should be ~5):", [len(u[1]) for u in wmunit_doc_list][:5])
        print("units_with_two_yes:", len(awmunit))
        print("units_without_two_yes:", len(uawmunit))
        logger.info(f'wmunit filter with two yes: {aim_len}, aim_count: {aim_count}, uaim_len: {len(uawmunit)}')
        if aim_len > aim_count:
            random_wmunit = random.sample(awmunit, aim_count)
        elif aim_len < aim_count:

            random_wmunit = awmunit + random.sample(uawmunit, aim_count-len(awmunit))
        else:
            random_wmunit = awmunit 

        return random_wmunit
    
    aim_wmunit_list = filter_wmunit(wmunit_doc_list, count_doc, aim_count)
    save_json(aim_wmunit_list , loc_path)
    # exit()
    count_doc = len(aim_wmunit_list)
    loc_list_sum = 0
    
    for i in range(count_doc):
        visiter.wm_unit = aim_wmunit_list[i][0]
        rag_document = []
        rag_document = [j for j in range(len(aim_wmunit_list[i][1])) if aim_wmunit_list[i][2][j] == 1 and aim_wmunit_list[i][3][j] == 1]

        if len( rag_document ) == 0: 
            rag_document= [j for j in range(len(aim_wmunit_list[i][1])) if aim_wmunit_list[i][2][j] == 1 ] 
            if len( rag_document ) == 0: 
                rag_document= [j for j in range(len(aim_wmunit_list[i][1])) if aim_wmunit_list[i][3][j] == 1 ]  
            else:
                rag_document = rag_document[:1]
            if len( rag_document ) == 0:
                rag_document.append(0)
            else:
                rag_document = rag_document[:1]

        try:
            visiter.rag_document = [aim_wmunit_list[i][1][j] for j in rag_document ]
        except Exception as e:
                print(f'e: {e}, rag_document: {rag_document}')
                logger.info(f"Error encountered: {e}. ")

                continue  # Continue loop after catching exception
        
        doc_info =    [ [aim_wmunit_list[i][1][j], aim_wmunit_list[i][2][j], aim_wmunit_list[i][3][j]] for j in  rag_document ]
        logger.info(f"doc_info: {doc_info}")
        
        # visiter.rag_document = wmunit_doc[3] ## is list
        loc_list = visiter.inject_wm()
        # logger.info(f'inject wmunit_doc[0] {wmunit_doc[0]}; loc_list {loc_list}, doc_info {doc_info}')
        # direct_check_list = 
        wmunit_inject_loc.append([visiter.wm_unit, loc_list, doc_info])
        loc_list_sum += len(loc_list)
        save_json(wmunit_inject_loc, loc_path)

    logger.info(f'inject watermark unit count: {count_doc}')
    logger.info(f'inject watermark len: {loc_list_sum}')
    # logger.info(f' inject watermark len : {count_doc}')
    return  True

import random


def wm_verify_run(verify_path, loc_path, doc_path, stat_path,  filter_flag=False, verify_num = 30):
    """
    Verify watermarks against a (possibly watermarked) RAG system.
    Builds a record of lookups and LLM decisions for analysis.
    """
    # verify_ids: [ [wm_unit(e1,e2,r)], [ids, WT_exist, WE], [result_flag, raw_result] ]
    verify_ids = []
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')

    wmunit_loc_list = []
    try:
        wmunit_loc_list = load_json(loc_path)
    except FileNotFoundError:
        logger.error(f'{loc_path} does not exist!')

    def find_loc(wmunit, ids):
        """Return (found_flag, index_in_loc_list) for a given (wmunit, ids) pair."""
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                for item, idsl in enumerate(wmunit_loc[1]):
                    if ids == idsl :
                        return 1, item
        return 0, -1
    
    def find_doc(wmunit, item):
        """Return the stored doc_info for a wmunit at the given loc index."""
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                return wmunit_loc[2][item] 

    eg0_verify_ids = [['e1','e2','r'],['ids','WT_exsit','WE'],['result','WD']]

    rwmunit_loc_list = random.sample(wmunit_loc_list, verify_num)

    for  wmunit_loc in rwmunit_loc_list:
        wmunit = wmunit_loc[0]
        if any(wmunit == vid[0] for vid in verify_ids):
            continue

        eg_verify_ids = copy.deepcopy(eg0_verify_ids)  # Use deep copy
        eg_verify_ids[0] = wmunit
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit

        rag_document, db_ids = visiter.ask_wm()
        db_search =  0
        wmunit_exist = -1
        wmunit_WE = -1
        wmunit_doc_info  = []
        db_search, item = find_loc(wmunit, db_ids)
        wmunit_WE = 0 if find_substrings_containing_sentences(rag_document, wmunit[:2])[1] == 0 else 1

        if item != -1:
            print(item)
            wmunit_doc_info = find_doc(wmunit, item)
            wmunit_exist = 0 if find_substrings_containing_sentences(wmunit_doc_info[0], wmunit[:2])[1] == 0 else 1
             
        if filter_flag:
            checker.rag_document = find_substrings_containing_sentences(rag_document, wmunit[:2])[0]
        else:
            checker.rag_document = rag_document

        eg_verify_ids[1] = [db_ids, db_search, wmunit_exist, checker.rag_document, wmunit_WE, wmunit_doc_info]
        result = checker.check_wm()
        result_flag = -1

        # Normalize yes/no/unknown → 1/0/2
        if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]:
            result_flag = 1
        elif result in ['No', 'no', 'NO', 'No.','no.','NO.']:
            result_flag = 0
        else:
            result_flag = 2
        
        eg_verify_ids[2] = [result_flag, result]

        if eg_verify_ids not in verify_ids:  # Ensure uniqueness
            verify_ids.append(eg_verify_ids)

        save_json(verify_ids, verify_path)

    logger.info(f'verify watermark doc len :{len(verify_ids)}')
    logger.info(f"[VERIFY] collection = {getattr(visiter.vectorstore.collection, 'name', None)}, count = {getattr(visiter.vectorstore.collection, 'count', lambda: None)()}")
    
    stat_result(verify_path,stat_path )
    return  True



def wm_verify_clean(verify_path, loc_path, doc_path, stat_path,  filter_flag=False, verify_num = 30):
    """
    Verify watermarks against a CLEAN RAG system (no injected watermarks).
    Useful for false-positive analysis.
    """
    # verify_ids: [ [wm_unit(e1,e2,r)], [ids, WT_exist, WE], [result_flag, raw_result] ]
    verify_ids = []
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')

    wmunit_loc_list = []
    try:
        wmunit_loc_list = load_json(loc_path)
    except FileNotFoundError:
        logger.error(f'{loc_path} does not exist!')

    def find_loc(wmunit, ids):
        """Return (found_flag, index_in_loc_list) for a given (wmunit, ids) pair."""
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                for item, idsl in enumerate(wmunit_loc[1]):
                    if ids == idsl :
                        return 1, item
        return 0, -1
    
    def find_doc(wmunit, item):
        """Return the stored doc_info for a wmunit at the given loc index."""
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                return wmunit_loc[2][item] 

    eg0_verify_ids = [['e1','e2','r'],['ids','WT_exsit','WE'],['result','WD']]

    rwmunit_loc_list = random.sample(wmunit_loc_list, verify_num)

    for  wmunit_loc in rwmunit_loc_list:
        wmunit = wmunit_loc[0]
        if any(wmunit == vid[0] for vid in verify_ids):
            continue

        eg_verify_ids = copy.deepcopy(eg0_verify_ids)  # Use deep copy
        eg_verify_ids[0] = wmunit
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit

        rag_document, db_ids = visiter.ask_wm()
        db_search =  0
        wmunit_exist = -1
        wmunit_WE = -1
        wmunit_doc_info  = []
        db_search, item = find_loc(wmunit, db_ids)
        wmunit_WE = 0 if find_substrings_containing_sentences(rag_document, wmunit[:2])[1] == 0 else 1

        if item != -1:
            print(item)
            wmunit_doc_info = find_doc(wmunit, item)
            wmunit_exist = 0 if find_substrings_containing_sentences(wmunit_doc_info[0], wmunit[:2])[1] == 0 else 1
             
        if filter_flag:
            checker.rag_document = find_substrings_containing_sentences(rag_document, wmunit[:2])[0]
        else:
            checker.rag_document = rag_document

        eg_verify_ids[1] = [db_ids, db_search, wmunit_exist, checker.rag_document, wmunit_WE, wmunit_doc_info]
        result = checker.check_wm()
        result_flag = -1

        # Normalize yes/no/unknown → 1/0/2
        if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.' ]:
            result_flag = 1
        elif result in ['No', 'no', 'NO', 'No.','no.','NO.']:
            result_flag = 0
        else:
            result_flag = 2
        
        eg_verify_ids[2] = [result_flag, result]

        if eg_verify_ids not in verify_ids:  # Ensure uniqueness
            verify_ids.append(eg_verify_ids)

        save_json(verify_ids, verify_path)

    logger.info(f'verify path :{len(verify_path)}')
    logger.info(f'verify watermark doc len :{len(verify_ids)}')
    logger.info(f"[VERIFY] collection = {getattr(visiter.vectorstore.collection, 'name', None)}, count = {getattr(visiter.vectorstore.collection, 'count', lambda: None)()}")
    
    stat_result(verify_path,stat_path )
    return  True

def stat_result(verify_path, stat_path):
    """
    Aggregate verification results and write a summary JSON to `stat_path`.
    Also prints a human-readable one-line summary.
    """
    verify_ids = []
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')

    total = 0
    correct = 0
    wrong = 0
    unknow = 0 
    db_search =  0
    wmunit_exist = 0
    wmunit_WE = 0 
 
    # logger.info(f"verify_ids: {verify_ids}")
    for vids in verify_ids:
        total += 1
        # logger.info(f"db_search: {db_search}, vids[1][1]: {vids[1][1]}")
        db_search += vids[1][1]
        wmunit_exist += vids[1][2]
        wmunit_WE += vids[1][4]
        if vids[2][0] == 0:
            wrong += 1
        elif vids[2][0]  == 1:
            correct += 1
        elif vids[2][0]  == 2:
            unknow += 1

    stat_dict = {}
    stat_dict['total'] = total
    stat_dict['correct'] = correct/total 
    stat_dict['wrong'] = wrong/total
    stat_dict['unknow'] = unknow/total
    stat_dict['db_search'] = db_search/total
    stat_dict['wmunit_exist'] = wmunit_exist/total
    stat_dict['wmunit_WE'] = wmunit_WE/total
    print(f"total-{total}, correct-{correct}-{correct/total:.2f}, wrong-{wrong}-{wrong/total:.2f}, \
                unknow-{unknow}-{unknow/total:.2f} db_search-{db_search}-{db_search/total:.2f},  \
                wmunit_exist-{wmunit_exist}-{wmunit_exist/total:.2f}, wmunit_WE-{wmunit_WE}-{wmunit_WE/total:.2f}")
    logger.info(f"total-{total}, correct-{correct}-{correct/total:.2f}, wrong-{wrong}-{wrong/total:.2f}, \
                unknow-{unknow}-{unknow/total:.2f} db_search-{db_search}-{db_search/total:.2f},  \
                wmunit_exist-{wmunit_exist}-{wmunit_exist/total:.2f}, wmunit_WE-{wmunit_WE}-{wmunit_WE/total:.2f}")
    save_json(stat_dict, stat_path)


# run_qa
# QA Prompt
RAG_QA_PROMPT = """You are a helpful assistant.
Use ONLY the following context to answer the user's question concisely.
If the answer cannot be found in the context, say you don't know.

[Context]
{context}

[Question]
{question}

[Answer]
"""

import csv, json
from typing import List, Dict, Iterable

def _ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def load_qid_list_txt(path: str):
    """Load a list of qids from a plain text file (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_qrels_query_ids(qrels_tsv_path: str) -> List[str]:
    """Load unique query-ids from a qrels TSV while preserving order."""
    seen = set()
    qids: List[str] = []
    with open(qrels_tsv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if i == 0 and parts[0].lower() in ("query-id", "query_id", "qid"):
                # Skip header
                continue
            qid = str(parts[0])
            if qid not in seen:
                seen.add(qid)
                qids.append(qid)
    return qids

def load_queries_subset_from_jsonl(queries_jsonl: str, wanted_ids: Iterable[str]) -> List[Dict]:
    """From queries.jsonl, keep only entries matching `wanted_ids` (order preserved)."""
    wanted = [str(x) for x in wanted_ids]
    pos = {qid: i for i, qid in enumerate(wanted)}  # preserve input order
    keep: Dict[str, Dict] = {}
    with open(queries_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id") or obj.get("id") or obj.get("qid") or obj.get("query_id"))
            if qid in pos:
                qtext = obj.get("text") or obj.get("query") or obj.get("question")
                keep[qid] = {"id": qid, "text": str(qtext)}
    # Reorder based on the original `wanted` list
    return [keep[qid] for qid in wanted if qid in keep]



from typing import List, Tuple, Dict, Any

def _map_internal_to_original_ids(vs: VectorStore, internal_ids: list[str]) -> list[str]:
    """
    Convert internal vector DB IDs to BEIR original corpus-ids.
    Uses whichever of 'doc_id' / 'id' / '_id' exists in metadata.
    """
    orig_ids = []
    try:
        got = vs.collection.get(ids=internal_ids, include=["metadatas"])
        metas = got.get("metadatas", []) or []
        for m in metas:
            m = m or {}
            # Priority: doc_id > id > _id
            oid = m.get("doc_id") or m.get("id") or m.get("_id")
            orig_ids.append(str(oid) if oid else None)
    except Exception:
        orig_ids = [None] * len(internal_ids)

    # If not found, return internal IDs (helpful for debugging)
    return [o if o else i for o, i in zip(orig_ids, internal_ids)]

def _vectorstore_search_ids_docs(vs: VectorStore, question: str, k: int):
    """
    Returns: (original_ids, docs)
    - original_ids: BEIR corpus-ids aligned with qrels
    - docs: top-k document texts
    """
    # 1) Preferred: use search_rich if available
    if hasattr(vs, "search_rich"):
        rich = vs.search_rich(question, n_results=k)  # [{"doc_id","internal_id","title","text","score"}...]
        # Exclude results lacking doc_id (e.g., ephemeral injected docs) from evaluation IDs
        original_ids = [r["doc_id"] for r in rich if r.get("doc_id")]
        docs         = [r["text"]   for r in rich]
        return original_ids, docs

    # 2) Legacy path: use search_context/retrieve and map internal IDs to originals
    internal_ids, docs = [], []
    if hasattr(vs, "search_context"):
        res = vs.search_context(question, k)
        if isinstance(res, dict):
            internal_ids = (res.get("ids", [[]]) or [[]])[0]
            docs        = (res.get("documents", [[]]) or [[]])[0]
        elif isinstance(res, list) and res and isinstance(res[0], (list, tuple)) and len(res[0]) >= 2:
            internal_ids = [r[0] for r in res[:k]]
            docs         = [r[1] for r in res[:k]]
    elif hasattr(vs, "retrieve"):
        res = vs.retrieve(question, k)
        if isinstance(res, dict):
            internal_ids = res.get("ids", [])[:k]
            docs         = (res.get("docs") or res.get("documents") or [])[:k]
        elif isinstance(res, list) and res and isinstance(res[0], (list, tuple)):
            internal_ids = [r[0] for r in res[:k]]
            docs         = [r[1] for r in res[:k]]

    internal_ids = [str(x) for x in internal_ids]
    docs = [str(d) for d in docs]
    original_ids = _map_internal_to_original_ids(vs, internal_ids)
    return original_ids, docs


def retrieve_ids_and_context(vs: VectorStore, question: str, k: int) -> Tuple[List[str], str]:
    ids, docs = _vectorstore_search_ids_docs(vs, question, k)
    ctx = "\n\n".join(str(d) for d in docs)
    return ids, ctx


def run_ask_qa_with_visiter(dataset: str, split: str, visiter, top_k: int, basepath_main_task: str, mode: str, use_qrels_ids: bool = True):
    qpath = os.path.join('datasets', dataset, 'queries.jsonl')
    out_dir = os.path.join(basepath_main_task, dataset)
    out_path = os.path.join(out_dir, 'wm_main_task.json' if mode=='wm' else 'clean_main_task.json')
    _ensure_dir(out_path)

    # ----- MSMARCO-specific handling -----
    if dataset == "msmarco":
        qid_file = '/mnt/ssd/TSF/knot-main/datasets/msmarco_10pct/query_ids_train_10pct.txt'
        qids = load_qid_list_txt(qid_file)
        queries = load_queries_subset_from_jsonl(qpath, qids)

    # ----- Others (hotpotqa, trec-covid, etc.) -----
    elif use_qrels_ids:
        qrels_tsv = os.path.join("datasets", dataset, "qrels", f"{split}.tsv")
        qids = load_qrels_query_ids(qrels_tsv)
        queries = load_queries_subset_from_jsonl(qpath, qids)
    else:
        print("Error in run_ask_qa_with_visiter()")

    total = len(queries)
    results = []

    for i, q in enumerate(queries, 1):
        print(q)
        qid, qtext = q['id'], q['text']

        # ✅ Single retrieval for both corpus-ids and context texts
        retrieved_ids, topk_docs = _vectorstore_search_ids_docs(visiter.vectorstore, qtext, k=top_k)
        ctx_str = "\n\n".join(topk_docs)

        # LLM call (preserve existing pipeline behavior)
        context_response, db_ids_internal, answer = visiter.ask_without_wm(qtext, k=top_k)

        # (Optional) extract scores if available
        try:
            scores_from_call = context_response.get('distances', [])[0] or context_response.get('scores', [])[0]
        except Exception:
            scores_from_call = []

        results.append({
            "query_id": qid,
            "query": qtext,
            "top_k": top_k,
            "retrieved_ids": [str(x) for x in retrieved_ids],
            "db_ids_internal": db_ids_internal,
            "context_texts": topk_docs,
            "retrieved_scores": scores_from_call,
            "llm_text": answer
        })

        # Progress logging
        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] queries processed...")

        if i % 20 == 0:
            save_json(results, out_path)

    save_json(results, out_path)
    print(f"[DONE] Saved results for {dataset} → {out_path}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever","contriever-msmarco","ance"])
    parser.add_argument('--eval_dataset', type=str, default="trec-covid", help='BEIR dataset to evaluate', choices= ['trec-covid','nq', 'msmarco', 'hotpotqa', 'nfcorpus'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--score_function', type=str, default='cosine', choices=['cosine','l2','ip' ])

    parser.add_argument('--top_k', type=int, default=5)

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name_rllm', default='gpt3.5', type=str, choices=['gpt3.5', 'claude', 'gemini', 'llama', 'vicuna', 'mistral'])
    parser.add_argument('--model_name_llm', type=str, default='gpt3.5')
    

    parser.add_argument('--adv_per_wmunit', type=int, default=5, help='Number of adversarial texts to generate per target query.')

    parser.add_argument('--gpu_id', type=int, default=0)

 
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    # Run switches
    parser.add_argument('--mutual_times', type=int, default=0, help='Number of interaction rounds; 0 means use LLM output directly') 
    parser.add_argument('--doc', type=int, default=0, help='Generate k documents for a given watermark unit')
    parser.add_argument('--inject', type=int, default=0, help='Inject documents into the dataset')
    parser.add_argument('--verify', type=int, default=0, help='Verify watermark units according to prepared list')
    parser.add_argument('--stat', type=int, default=0, help='Aggregate and save statistical verification information')
    parser.add_argument('--clean', type=int, default=0, help='Verify watermark units under a clean RAG (no injection)')

    # ask_qa
    parser.add_argument('--ask_qa_wm', type=int, default=0,
                        help='If 1, run QA over datasets/{dataset}/queries.jsonl on the (possibly watermarked) DB and save to output/main_task/{dataset}/wm_main_task.json')
    parser.add_argument('--ask_qa_clean', type=int, default=0,
                        help='If 1, run QA over datasets/{dataset}/queries.jsonl on the clean DB and save to output/main_task/{dataset}/clean_main_task.json')
    parser.add_argument('--qa_top_k', type=int, default=10,
                        help='Top-k for retrieval when running ask_qa=1 (default: 10)')
    parser.add_argument("--use_qrels_ids", action="store_true",
                        help="Use only query-ids from qrels/<split>.tsv (except msmarco)")

    # Save paths
    parser.add_argument('--basepath', type=str, default='/mnt/ssd/TSF/knot-main/output', help='Root for watermark_doc, inject_info, verify_info, stat_results')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    watermark_unit_path = os.path.join(args.basepath, 'wm_prepare', args.eval_dataset, 'wmunit.json')
    basepath = os.path.join(args.basepath,'wm_generate',args.eval_dataset)

    LOG_FILE = os.path.join(basepath,args.model_name_rllm,str(args.mutual_times),'log.log')
    file_exist(LOG_FILE)

    logger = Log(log_file=LOG_FILE).get(__file__)

    logger.info(f'args:{args}')
    print(LOG_FILE)

    # exit()
    wmunit_doc_path = os.path.join(basepath,str(args.mutual_times),'wmuint_doc.json')
    wmunit_inject_path = os.path.join(basepath,str(args.mutual_times),'wmuint_inject.json')

     
    args.model_config_path = f'model_configs/{args.model_name_llm}_config.json'
    logger.info(f'llm args.model_config_path: {args.model_config_path}')
    llm = create_model(args.model_config_path)
     
    args.model_config_path = f'model_configs/{args.model_name_rllm}_config.json'
    logger.info(f'rllm args.model_config_path: {args.model_config_path}')
    rllm = create_model(args.model_config_path)
    query_prompt = '1 +2 = ?'

 
    wmunit_verify_path = os.path.join(basepath,args.model_name_rllm, str(args.mutual_times), 'wmuint_verify.json')
    wmunit_stat_path = os.path.join(basepath, args.model_name_rllm,str(args.mutual_times), 'wmuint_stat.json')
        
 
    advisor = Advisor(llm)
    
    # Create vector DB
    collection_name = args.eval_dataset+'_'+args.eval_model_code+'_'+args.score_function
    logger.info(f"Collection_name: {collection_name}")
    # exit()

    # Load retriever model
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    
    # Load target queries and answers
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
    else:
        corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    datalen = len(corpus)
    collection_exist, collection_len = check_collection(collection_name)
    logger.info(f'datalen:{datalen}, collection_len:{collection_len}')
    # exit()
    if  collection_exist and datalen==collection_len:
        use_local = True
    else :
        # use_local = False
        logger.info(f'please run rag/vectorstore.py to create vectorstore for {collection_name} ')
        exit()

    vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local=True)
    
    visiter = Visiter(llm, rllm, vectorstore)
    
    checker = Checker(llm)
    
    if args.clean == 1:
        wm_verify_clean(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path, verify_path=wmunit_verify_path,stat_path=wmunit_stat_path)
    if args.doc == 1:
        doc_llm_run_mutual(doc_path=wmunit_doc_path, mutual_times=args.mutual_times)
    if args.inject == 1:
        wm_inject_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path)

    filter_flag = False
    if args.verify == 1:
        wm_verify_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path, verify_path=wmunit_verify_path,stat_path=wmunit_stat_path, filter_flag=False)
        stat_result( verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)

    # === (New) ask_qa: run queries.jsonl → wm_main_task.json ===
    if args.ask_qa_wm == 1:
        basepath_main_task = os.path.join(args.basepath, 'main_task')
        out_path = run_ask_qa_with_visiter(
            dataset=args.eval_dataset,
            split=args.split,
            visiter=visiter,
            top_k=args.qa_top_k,
            basepath_main_task=basepath_main_task,
            mode='wm'
        )
        logger.info(f'ask_qa saved to: {out_path}')

    if args.ask_qa_clean == 1:
        vectorstore.clean_collect()
        basepath_main_task = os.path.join(args.basepath, 'main_task')
        out_path = run_ask_qa_with_visiter(
            dataset=args.eval_dataset,
            split=args.split,
            visiter=visiter,
            top_k=args.qa_top_k,
            basepath_main_task=basepath_main_task,
            mode='clean'
        )
        logger.info(f'ask_qa saved to: {out_path}')




"""
# generation
python src/main_QA.py --eval_dataset 'trec-covid' --eval_model_code "contriever" --score_function 'cosine' --doc 1 --inject 0 --verify 0 --stat 0 --mutual_times 10
# injection
python src/main_QA.py --eval_dataset 'trec-covid' --eval_model_code "contriever" --score_function 'cosine' --doc 0 --inject 1 --verify 0 --stat 0 --mutual_times 10
# verification
python src/main_QA.py --eval_dataset 'trec-covid' --eval_model_code "contriever" --score_function 'cosine' --doc 0 --inject 0 --verify 1 --stat 1 --mutual_times 10
python src/main_QA.py --eval_dataset 'nq' --eval_model_code "contriever" --score_function 'cosine' --doc 0 --inject 0 --verify 1 --stat 1 --mutual_times 10
# clean
python src/main_QA.py --eval_dataset 'trec-covid' --eval_model_code "contriever" --score_function 'cosine' --doc 0 --inject 0  --clean 1 --stat 1 --mutual_times 10


# ask_qa (query a watermarked DB) → output/main_task/trec-covid/wm_main_task.json
python src/main_QA.py --eval_dataset 'trec-covid' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "llama" --ask_qa_clean 1 --top_k 10 --qa_top_k 10
python src/main_QA.py --eval_dataset 'trec-covid' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "llama" --ask_qa_wm 1 --top_k 10 --qa_top_k 10

python src/main_QA.py --eval_dataset 'trec-covid' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "gpt3.5" --ask_qa_wm 1 --top_k 10 --qa_top_k 10 
python src/main_QA.py --eval_dataset 'nfcorpus' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "gpt3.5" --ask_qa_wm 1 --top_k 10 --qa_top_k 10 
python src/main_QA.py --eval_dataset 'nq' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "gpt3.5" --ask_qa_wm 1 --top_k 10 --qa_top_k 10 
python src/main_QA.py --eval_dataset 'hotpotqa' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "gpt3.5" --ask_qa_wm 1 --top_k 10 --qa_top_k 10 
python src/main_QA.py --eval_dataset 'msmarco' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "gpt3.5" --ask_qa_wm 1 --top_k 10 --qa_top_k 10 



# injection
python src/main_QA.py --eval_dataset 'hotpotqa' --eval_model_code "contriever" --score_function 'cosine' --doc 0 --inject 1 --verify 0 --stat 0 --mutual_times 10
# ask_qa (query a watermarked DB) → output/main_task/trec-covid/wm_main_task.json
python src/main_QA.py --eval_dataset 'hotpotqa' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "llama" --ask_qa_wm 1 --top_k 10 --qa_top_k 10

# injection
python src/main_QA.py --eval_dataset 'msmarco' --eval_model_code "contriever" --score_function 'cosine' --doc 0 --inject 1 --verify 0 --stat 0 --mutual_times 10
# ask_qa (query a watermarked DB) → output/main_task/trec-covid/wm_main_task.json
python src/main_QA.py --eval_dataset 'msmarco' --eval_model_code "contriever" --score_function 'cosine' --model_name_rllm "llama" --ask_qa_wm 1 --top_k 10 --qa_top_k 10
"""
