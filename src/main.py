import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add project root to sys.path
sys.path.append(project_root)
import argparse
 
from watermark_role import Advisor, Visiter, Checker 

from rag.vectorstore import VectorStore, check_collection
from src.utils import load_json, save_json, find_substrings_containing_sentences, Log, extract_doc_list, extract_doc, file_exist
from src.models import create_model
from src.utils import load_beir_datasets, load_models

import re
import time
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
# logger = Log().get(__file__)

def doc_llm_run_single(doc_path):
    """
    Single-LLM generation pipeline (no multi-LLM interaction).
    - Generation: use only advisor.get_document() (no feedback/interaction)
    - Verification: reuse direct_check / simulate_check logic
    Save format: [wmunit, WT_list, WT_direct_flag, WT_simulate_flag]
    """
    wmunit_list = load_json(watermark_unit_path)

    # Append to existing result file if it already exists
    try:
        wmunit_doc_list = load_json(doc_path)
    except FileNotFoundError:
        logger.warning(f'{doc_path} does not exist! New file will be created.')
        wmunit_doc_list = []

    # Number of texts to generate per watermark unit
    advisor.K = args.adv_per_wmunit

    def get_WT_list_single(wmunit):
        """Collect K WTs using only the advisor (no interaction)."""
        advisor.wm_unit = wmunit
        WT_list, count = [], 0
        while len(WT_list) != advisor.K and count < 10:
            count += 1
            try:
                WTs = advisor.get_document()  # single LLM call
                if isinstance(WTs, str):
                    WT_list = extract_doc_list(WTs)
                    logger.info(f'[single] try#{count} | {wmunit} -> {len(WT_list)} items')
            except Exception as e:
                logger.info(f'[single] error: {e}; retrying...')
                continue
        return WT_list

    def direct_check(wmunit, WT_list):
        flags = []
        for WT in WT_list:
            checker.rag_document = WT
            checker.wm_unit = wmunit
            res = checker.check_wm()
            flags.append(1 if str(res).strip('.').lower() == 'yes'
                         else (0 if str(res).strip('.').lower() == 'no' else 2))
        return flags

    def simulate_check(wmunit, WT_list):
        flags = []
        for WT in WT_list:
            WE = visiter.ask_wm_with_wt(WT)  # validate once more with a simulated response
            checker.rag_document = WE
            checker.wm_unit = wmunit
            res = checker.check_wm()
            flags.append(1 if str(res).strip('.').lower() == 'yes'
                         else (0 if str(res).strip('.').lower() == 'no' else 2))
        return flags

    logger.info(f'[single] total watermark units: {len(wmunit_list)}')
    for idx, wmunit in enumerate(wmunit_list):
        # Skip units that have already been processed
        if any(wmunit == row[0] for row in wmunit_doc_list):
            logger.info(f'[single] skip existing unit #{idx}: {wmunit}')
            continue

        logger.info(f'[single] process unit #{idx}: {wmunit}')
        WT_list = get_WT_list_single(wmunit)  # no interaction

        WT_direct_flag = direct_check(wmunit, WT_list)
        WT_simulate_flag = simulate_check(wmunit, WT_list)

        wmunit_doc = [wmunit, WT_list, WT_direct_flag, WT_simulate_flag]
        wmunit_doc_list.append(wmunit_doc)

        logger.info(f'[single] save -> {doc_path}')
        save_json(wmunit_doc_list, doc_path)

    save_json(wmunit_doc_list, doc_path)
    return True


def doc_llm_run_mutual(doc_path, mutual_times=0):
    """
    Example:
        python wm_generate/role.py --doc 1 --auto 2
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
            count += 1
            try:
                if isinstance(WTs, str):
                    logger.info(f'count: {count} ,WTs : {WTs}')
                    WT_list = extract_doc_list(WTs)
                    len_WT = len(WT_list)
                    logger.info(f'count: {count} ,{wmunit} len_WT {len_WT}; {WTs}; {WT_list}')
            except Exception as e:
                print(f"Error encountered: {e}. Retrying...")
                continue  # continue after catching exception
        return WT_list

    def direct_check(wmunit, WT_list):
        WT_flag = []
        for WT in WT_list:
            checker.rag_document = WT
            checker.wm_unit = wmunit
            result = checker.check_wm()
    
            if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.']:
                WT_flag.append(1)
            elif result in ['No', 'no', 'NO', 'No.', 'no.', 'NO.']:
                WT_flag.append(0)
            else:
                WT_flag.append(2)
        return WT_flag
    
    def simulate_check(wmunit, WT_list):
        WT_flag = []
        for WT in WT_list:
            WE = visiter.ask_wm_with_wt(WT)
            checker.rag_document = WE
            checker.wm_unit = wmunit
            result = checker.check_wm()
    
            if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.']:
                WT_flag.append(1)
            elif result in ['No', 'no', 'NO', 'No.', 'no.', 'NO.']:
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
            true_list = ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.']
            while WD1 not in true_list or WD2 not in true_list or WT in WT_list_interaction:
                if flag > mtimes:
                    break
                flag += 1
                logger.info(f'mutual_time: {flag}, WD1:{WD1}, WD2:{WD2}, WT:{WT}, WE:{WE}')
                try:
                    WT = advisor.get_document_feedback_both(WT=WT, WE=WE, WD1=WD1, WD2=WD2)
                    logger.info(f'before WT for advisor with feedback: {WT}')
                    WT = extract_doc(WT)
                    logger.info(f'after WT for advisor with feedback: {WT}')
                    checker.rag_document = WT
                    WD1 = checker.check_wm()

                    WE = visiter.ask_wm_with_wt(WT)
                    checker.rag_document = WE
                    WD2 = checker.check_wm()
                except Exception as e:
                    logger.info(f"Error encountered: {e}. Retrying...")
                    continue  # continue after catching exception

            WT_list_interaction.append(WT)
        logger.info(f'WT_list_interaction: {WT_list_interaction}')
        return WT_list_interaction
 
    logger.info(f"len wmunit: {len(wmunit_list)}")
    for item, wmunit in enumerate(wmunit_list):
        if any(wmunit == wl[0] for wl in wmunit_doc_list):
            print(item, wmunit)
            continue
        logger.info(f'item:{item}, wmunit:{wmunit}')
        wmunit_doc = []
        WT_list = get_WT_list(wmunit)

        if mutual_times != 0:
            WT_list = get_WT_interation_both_list(wmunit, WT_list, mutual_times)

        WT_direct_flag = direct_check(wmunit, WT_list)
        WT_simulate_flag = simulate_check(wmunit, WT_list)

        wmunit_doc.append(wmunit) 
        wmunit_doc.append(WT_list)
        wmunit_doc.append(WT_direct_flag)
        wmunit_doc.append(WT_simulate_flag)
        # if wmunit_doc not in wmunit_doc_list:
        wmunit_doc_list.append(wmunit_doc)
        logger.info(f'save path :{doc_path}')
        save_json(wmunit_doc_list, doc_path)
        
    wmdoclen = len(wmunit_doc_list)
    wmlen = len(wmunit_list)
    # logger.info(f'wm len {wmlen}: wm doc len {wmdoclen}')
    # assert(wmlen == wmdoclen)
    
    save_json(wmunit_doc_list, doc_path)

    return True


import random

def wm_inject_run(doc_path, loc_path, aim_count=50, seed=42):
    """
    Inject exactly one text per block (watermark unit).
    Priority: (checker2==1 & checker3==1) > checker2==1 > checker3==1 > index 0.
    """
    if seed is not None:
        random.seed(seed)

    wmunit_doc_list = load_json(doc_path)
    wmunit_inject_loc = []

    def filter_wmunit(wmunit_doc_list, aim_count):
        """Prefer blocks where both checkers are 1; fix the selection to aim_count (default 50)."""
        awmunit, uawmunit = [], []
        for item in wmunit_doc_list:
            # expected item structure: [wm_unit, texts, checker2_list, checker3_list, ...]
            c2 = item[2]
            c3 = item[3]
            both_yes = [j for j in range(len(item[1])) if c2[j] == 1 and c3[j] == 1]
            if len(both_yes) == 0:
                uawmunit.append(item)
            else:
                awmunit.append(item)

        logger.info(f'wmunit filter with two yes: {len(awmunit)}, aim_count: {aim_count}')

        if len(awmunit) > aim_count:
            selected = random.sample(awmunit, aim_count)
        elif len(awmunit) < aim_count:
            need = aim_count - len(awmunit)
            selected = awmunit + random.sample(uawmunit, need)
        else:
            selected = awmunit
        return selected

    # 1) select aim_count blocks
    aim_wmunit_list = filter_wmunit(wmunit_doc_list, aim_count)
    save_json(aim_wmunit_list, loc_path)  # snapshot of the 50 selected blocks

    injected_blocks = 0
    injected_texts = 0

    # 2) choose exactly one document index per block and inject
    for i, item in enumerate(aim_wmunit_list):
        wm_unit = item[0]
        texts = item[1]
        c2 = item[2]
        c3 = item[3]

        # candidate sets
        both_yes = [j for j in range(len(texts)) if c2[j] == 1 and c3[j] == 1]
        c2_only = [j for j in range(len(texts)) if c2[j] == 1 and not (c2[j] == 1 and c3[j] == 1)]
        c3_only = [j for j in range(len(texts)) if c3[j] == 1 and not (c2[j] == 1 and c3[j] == 1)]

        # Priority selection (uncomment to enforce priority choice)
        # if len(both_yes) > 0:
        #     chosen_idx = random.choice(both_yes)
        # elif len(c2_only) > 0:
        #     chosen_idx = random.choice(c2_only)
        # elif len(c3_only) > 0:
        #     chosen_idx = random.choice(c3_only)
        # else:
        chosen_idx = 0  # safe default when all checks fail

        # configure visitor and inject
        visiter.wm_unit = wm_unit
        try:
            visiter.rag_document = texts[chosen_idx]  # exactly one
        except Exception as e:
            logger.info(f"Error encountered: {e}. Skip this block.")
            continue

        doc_info = [[texts[chosen_idx], c2[chosen_idx], c3[chosen_idx]]]

        loc_list = visiter.inject_wm()  # actual injection
        wmunit_inject_loc.append([wm_unit, loc_list, doc_info])
        save_json(wmunit_inject_loc, loc_path)  # incremental save

        injected_blocks += 1
        injected_texts += 1  # one per block

    logger.info(f'inject watermark unit blocks: {injected_blocks} (target {aim_count})')
    logger.info(f'injected texts (should equal blocks): {injected_texts}')

    return True


def wm_verify_run(verify_path, loc_path, doc_path, stat_path, answer_path, filter_flag=False, verify_num=30):
    """
    Verify watermark presence and store both verification trace and answers.
    """
    # verify_ids format:
    #   [ wmunit, [db_ids, db_search, wmunit_exist, checker_input, wmunit_WE, wmunit_doc_info], [result_flag, raw_result] ]
    verify_ids = []
    wm_answer_records = []

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
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                for item, idsl in enumerate(wmunit_loc[1]):
                    if ids == idsl:
                        return 1, item
        return 0, -1
    
    def find_doc(wmunit, item):
        for wmunit_loc in wmunit_loc_list:
            if wmunit == wmunit_loc[0]:
                return wmunit_loc[2][item]

    eg0_verify_ids = [['e1', 'e2', 'r'], ['ids', 'WT_exist', 'WE'], ['result', 'WD']]

    rwmunit_loc_list = random.sample(wmunit_loc_list, verify_num)

    for wmunit_loc in rwmunit_loc_list:
        wmunit = wmunit_loc[0]
        if any(wmunit == vid[0] for vid in verify_ids):
            continue

        eg_verify_ids = copy.deepcopy(eg0_verify_ids)  # deep copy
        eg_verify_ids[0] = wmunit
        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit

        # retrieve and verify
        # rag_document, db_ids = visiter.ask_wm()
        rag_document, db_ids = visiter.ask_wm_k(k=30)
        db_search = 0
        wmunit_exist = -1
        wmunit_WE = -1
        wmunit_doc_info = []
        db_search, item = find_loc(wmunit, db_ids)
        wmunit_WE = 0 if find_substrings_containing_sentences(rag_document, wmunit[:2])[1] == 0 else 1

        if item != -1:
            print(item)
            wmunit_doc_info = find_doc(wmunit, item)
            # defensive handling for list/None
            doc_text = wmunit_doc_info[0] if wmunit_doc_info else ""
            if isinstance(doc_text, list):
                doc_text = " ".join(map(str, doc_text))
            elif doc_text is None:
                doc_text = ""
            else:
                doc_text = str(doc_text)
            wmunit_exist = 0 if find_substrings_containing_sentences(doc_text, wmunit[:2])[1] == 0 else 1
             
        if filter_flag:
            checker.rag_document = find_substrings_containing_sentences(rag_document, wmunit[:2])[0]
        else:
            checker.rag_document = rag_document

        eg_verify_ids[1] = [db_ids, db_search, wmunit_exist, checker.rag_document, wmunit_WE, wmunit_doc_info]
        result = checker.check_wm()
        result_flag = -1

        # normalize yes/no/unknown → 1/0/2
        if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.']:
            result_flag = 1
        elif result in ['No', 'no', 'NO', 'No.', 'no.', 'NO.']:
            result_flag = 0
        else:
            result_flag = 2
        
        eg_verify_ids[2] = [result_flag, result]

        # Save rag_document + wmunit + tuple for downstream analyses
        wm_answer_record = {
            "wmunit": wmunit,
            "tuple": {"E1": wmunit[0], "R": wmunit[2], "E2": wmunit[1]},
            "rag_answer": rag_document,
            "result_flag": result_flag,
            "result": result
        }
        if wm_answer_record not in wm_answer_records:
            wm_answer_records.append(wm_answer_record)
        save_json(wm_answer_records, answer_path)

        # Save verification trace
        if eg_verify_ids not in verify_ids:
            verify_ids.append(eg_verify_ids)
        save_json(verify_ids, verify_path)

        if len(verify_ids) % 5 == 0:
            stat_result(verify_path, stat_path)

    stat_result(verify_path, stat_path)
    logger.info(f'verify watermark doc len :{len(verify_ids)}')
    return True


def wm_verify_clean(verify_path, loc_path, doc_path, stat_path, verify_num=30):
    """
    Verify on a clean setup (no watermark injection) for false-positive analysis.
    """
    verify_ids = []  # [wmunit, [db_ids, context_db, rag_document], [result, result_flag]]
    try:
        verify_ids = load_json(verify_path)
    except FileNotFoundError:
        logger.error(f'{verify_path} does not exist!')

    wmunit_doc_list = load_json(doc_path)

    rwmunit_loc_list = random.sample(wmunit_doc_list, verify_num)

    for wmunit_loc in rwmunit_loc_list:
        wmunit = wmunit_loc[0]
        if any(wmunit == vid[0] for vid in verify_ids):
            continue

        visiter.wm_unit = wmunit
        checker.wm_unit = wmunit

        rag_document, db_ids, context_db = visiter.ask_wm_test()
        checker.rag_document = rag_document
        result = checker.check_wm()
        result_flag = -1

        # normalize yes/no/unknown → 1/0/2
        if result in ['Yes', 'YES', 'yes', 'Yes.', 'YES.', 'yes.']:
            result_flag = 1
        elif result in ['No', 'no', 'NO', 'No.', 'no.', 'NO.']:
            result_flag = 0
        else:
            result_flag = 2
        
        eg_verify_ids = [wmunit, [db_ids, context_db, rag_document], [result, result_flag]]
        if eg_verify_ids not in verify_ids:  # ensure uniqueness
            verify_ids.append(eg_verify_ids)

        save_json(verify_ids, verify_path)

        if len(verify_ids) % 5 == 0:
            stat_result(verify_path, stat_path)

    stat_result(verify_path, stat_path)
    logger.info(f'verify watermark doc len :{len(verify_ids)}')
    return True


def stat_result(verify_path, stat_path):
    """
    Aggregate verification results and write stats to `stat_path`.
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
    db_search = 0
    wmunit_exist = 0
    wmunit_WE = 0 

    for vids in verify_ids:
        total += 1
        db_search += vids[1][1]
        wmunit_exist += vids[1][2]
        wmunit_WE += vids[1][4]
        if vids[2][0] == 0:
            wrong += 1
        elif vids[2][0] == 1:
            correct += 1
        elif vids[2][0] == 2:
            unknow += 1

    stat_dict = {
        'total': total,
        'correct': correct / total if total else 0.0,
        'wrong': wrong / total if total else 0.0,
        'unknow': unknow / total if total else 0.0,
        'db_search': db_search / total if total else 0.0,
        'wmunit_exist': wmunit_exist / total if total else 0.0,
        'wmunit_WE': wmunit_WE / total if total else 0.0,
    }
    print(f"total-{total}, correct-{correct}-{stat_dict['correct']:.2f}, wrong-{wrong}-{stat_dict['wrong']:.2f}, "
          f"unknow-{unknow}-{stat_dict['unknow']:.2f} db_search-{db_search}-{stat_dict['db_search']:.2f},  "
          f"wmunit_exist-{wmunit_exist}-{stat_dict['wmunit_exist']:.2f}, wmunit_WE-{wmunit_WE}-{stat_dict['wmunit_WE']:.2f}")
    logger.info(f"total-{total}, correct-{correct}-{stat_dict['correct']:.2f}, wrong-{wrong}-{stat_dict['wrong']:.2f}, "
                f"unknow-{unknow}-{stat_dict['unknow']:.2f} db_search-{db_search}-{stat_dict['db_search']:.2f},  "
                f"wmunit_exist-{wmunit_exist}-{stat_dict['wmunit_exist']:.2f}, wmunit_WE-{wmunit_WE}-{stat_dict['wmunit_WE']:.2f}")
    save_json(stat_dict, stat_path)


def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever",
                        choices=["contriever", "contriever-msmarco", "ance"])
    parser.add_argument('--eval_dataset', type=str, default="trec-covid",
                        help='BEIR dataset to evaluate',
                        choices=['trec-covid', 'nq', 'msmarco', 'hotpotqa', 'nfcorpus'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--score_function', type=str, default='cosine',
                        choices=['cosine', 'l2', 'ip'])

    parser.add_argument('--top_k', type=int, default=5)

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name_rllm', default='gpt3.5', type=str,
                        choices=['gpt3.5', 'llama', 'mistral', 'vicuna', 'claude', 'gemini'])
    parser.add_argument('--model_name_llm', type=str, default='gpt3.5')
    
    parser.add_argument('--adv_per_wmunit', type=int, default=1,
                        help='The number of adversarial texts for each target query.')

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    # Run switches
    parser.add_argument('--mutual_times', type=int, default=0,
                        help='Repeat several times for interaction; 0 means use LLM output directly')
    parser.add_argument('--doc', type=int, default=0, help='Generate k documents for a given wmunit')
    parser.add_argument('--inject', type=int, default=0, help='Inject documents into dataset')
    parser.add_argument('--verify', type=int, default=0, help='Verify according to wmunit list')
    parser.add_argument('--stat', type=int, default=0, help='Statistical verification information')
    parser.add_argument('--clean', type=int, default=0, help='Verify wmunit under a clean RAG')

    # Save paths (updated to knot-main)
    parser.add_argument('--basepath', type=str, default='/mnt/ssd/TSF/knot-main/output',
                        help='Save watermark_doc, inject_info, verify_info, stat_results')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    watermark_unit_path = os.path.join(args.basepath, 'wm_prepare', args.eval_dataset, 'wmunit.json')
    basepath = os.path.join(args.basepath, 'wm_generate', args.eval_dataset)

    LOG_FILE = os.path.join(basepath, args.model_name_rllm, str(args.mutual_times), 'log.log')
    file_exist(LOG_FILE)

    logger = Log(log_file=LOG_FILE).get(__file__)

    logger.info(f'args:{args}')
    print(LOG_FILE)

    # exit()
    wmunit_doc_path = os.path.join(basepath, str(args.mutual_times), 'wmuint_doc.json')
    wmunit_inject_path = os.path.join(basepath, str(args.mutual_times), 'wmuint_inject.json')

    args.model_config_path = f'model_configs/{args.model_name_llm}_config.json'
    logger.info(f'LLM args.model_config_path: {args.model_config_path}')
    llm = create_model(args.model_config_path)
     
    args.model_config_path = f'model_configs/{args.model_name_rllm}_config.json'
    logger.info(f'RLLM args.model_config_path: {args.model_config_path}')
    rllm = create_model(args.model_config_path)
    query_prompt = '1 +2 = ?'

    wmunit_answer_path = os.path.join(basepath, args.model_name_rllm, str(args.mutual_times), 'verify_answers.json')
    wmunit_verify_path = os.path.join(basepath, args.model_name_rllm, str(args.mutual_times), 'wmuint_verify.json')
    wmunit_stat_path = os.path.join(basepath, args.model_name_rllm, str(args.mutual_times), 'wmuint_stat.json')
        
    advisor = Advisor(llm)
    # response = advisor.get_document()
    # print(response)

    checker = Checker(llm)
    # response = checker.check_wm()
    # print(response)

    # Create vector DB
    collection_name = args.eval_dataset + '_' + args.eval_model_code + '_' + args.score_function
    print(collection_name)
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
    if collection_exist and datalen == collection_len:
        use_local = True
    else:
        # use_local = False
        logger.info(f'please run rag/vectorstore.py to create vectorstore for {collection_name} ')
        exit()

    vectorstore = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local=True)
    if args.inject == 1 or args.clean == 1:
        vectorstore.clean_collect()
    
    visiter = Visiter(llm, rllm, vectorstore)
    result = visiter.ask_wm()
    print(result)
    result = visiter.ask_wm_with_wt(visiter.rag_document)
    print(result)
    checker = Checker(llm)
    checker.rag_document = result
    response = checker.check_wm()
    print(response)

    if args.clean == 1:
        wm_verify_clean(doc_path=wmunit_doc_path, verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)

    if args.doc == 1:
        start_time = time.time()
        print("[INFO] doc_llm_run_mutual started...")

        # doc_llm_run_mutual(doc_path=wmunit_doc_path, mutual_times=args.mutual_times)
        doc_llm_run_single(doc_path=wmunit_doc_path)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[INFO] doc_llm_run_mutual finished. Elapsed time: {elapsed:.2f} seconds")

    if args.inject == 1:
        wm_inject_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path)

    filter_flag = False
    if args.verify == 1:
        wm_verify_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path,
                      verify_path=wmunit_verify_path, stat_path=wmunit_stat_path,
                      answer_path=wmunit_answer_path, filter_flag=False)

    if args.stat == 1:
        stat_result(verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)

    # filter_flag == True
    # wmunit_verify_path = os.path.join(basepath, args.model_name_rllm, str(args.mutual_times), 'wmuint_verify_filter.json')
    # wmunit_stat_path = os.path.join(basepath, args.model_name_rllm, str(args.mutual_times), 'wmuint_stat_filter.json')
    # if args.verify == 1:
    #     wm_verify_run(doc_path=wmunit_doc_path, loc_path=wmunit_inject_path, verify_path=wmunit_verify_path, stat_path=wmunit_stat_path, filter_flag=True)
    # if args.stat == 1:
    #     stat_result(verify_path=wmunit_verify_path, stat_path=wmunit_stat_path)


# Example runs:
# python src/main.py --eval_dataset "msmarco"  --eval_model_code "contriever" --score_function 'cosine' --mutual_times 10 --doc 1
# python src/main.py --eval_dataset "hotpotqa" --eval_model_code "contriever" --score_function 'cosine' --mutual_times 10 --doc 1 --inject 1 --verify 1 --stat 1
# For model 'llama7b':
# python src/main.py --eval_dataset "nfcorpus"  --eval_model_code "contriever" --score_function 'cosine' --mutual_times 10 --doc 0 --inject 0 --verify 1 --stat 1 --model_name_rllm 'llama7b'
# python src/main.py --eval_dataset "trec-covid" --eval_model_code "contriever" --score_function 'cosine' --mutual_times 10 --doc 0 --inject 0 --verify 1 --stat 1 --model_name_rllm 'llama7b' --gpu_id 3
