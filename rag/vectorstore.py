import argparse
import os
import sys
import torch
import chromadb

# Disable Chroma telemetry through environment variable
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
from chromadb.config import Settings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils import load_beir_datasets, load_models

ChromadbPath = './chromadb_db'

class VectorStore:
    def __init__(self, embedding_model, tokenizer, get_emb, dataset, device, collection_name, use_local, distance='cosine'):
        self.chroma_client = chromadb.PersistentClient(path=ChromadbPath, settings=Settings())
        collections = self.chroma_client.list_collections()
        collection_exists = any(col.name == collection_name for col in collections)

        if collection_exists and use_local:
            print(f"Using existing ChromaDB collection: {collection_name}")
            self.collection = self.chroma_client.get_collection(name=collection_name)
        else:
            if collection_exists:
                print(f"Deleting existing ChromaDB collection: {collection_name}")
                self.chroma_client.delete_collection(name=collection_name)
            print(f"Creating new ChromaDB collection: {collection_name}")
            self.collection = self.chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": distance})

        self.embedding_model = embedding_model.to(device).eval()
        self.tokenizer = tokenizer
        self.get_emb = get_emb
        self.dataset = dataset
        self.device = device

    def get_embedding(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.get_emb(self.embedding_model, inputs).squeeze().tolist()
        return emb

    # Search the ChromaDB collection for relevant context based on a query
    def search_context(self, query, n_results):
        text_emb = self.get_embedding(query)
        return self.collection.query(query_embeddings=text_emb, n_results=n_results)

    def populate_vectors(self):
        current_count = self.collection.count()
        for count, key in enumerate(sorted(self.dataset.keys())):
            if count < current_count:
                continue
            value = self.dataset[key]
            text_emb = self.get_embedding(value['text'])
            self.collection.add(
                embeddings=[text_emb],
                documents=[value['text']],
                ids=[f'id_{count}'],
                metadatas={'title': value['title'], 'id': key, 'change': False}
            )
            if count % 500 == 0:
                print(count)

    def update_context(self, doc_id, addition='', pos='end'):
        doc = self.collection.get(ids=[doc_id])
        meta = doc['metadatas'][0]
        if meta['change'] and not addition:
            return
        if not meta['change'] and not addition:
            return
        context = doc['documents'][0]
        updated_context = (context + ' ' + addition) if pos == 'end' else (addition + ' ' + context)
        new_emb = self.get_embedding(updated_context)
        self.collection.update(
            ids=[doc_id],
            embeddings=[new_emb],
            documents=[updated_context],
            metadatas={'title': meta['title'], 'id': meta['id'], 'change': True}
        )

    def clean_collect(self):
        docs = self.collection.get(where={'change': True})
        for i, doc_id in enumerate(docs['ids']):
            if int(doc_id.split('_')[1]) >= len(self.dataset) or docs['metadatas'][i]['id'] == ' ':
                self.collection.delete(doc_id)
            else:
                self.update_context(doc_id)

    def get_id(self, doc_id):
        return self.collection.get(ids=[doc_id])

    def inject_direct(self, text):
        emb = self.get_embedding(text)
        doc_id = self.collection.count()
        self.collection.add(
            embeddings=[emb],
            documents=[text],
            ids=[f'id_{doc_id}'],
            metadatas={'title': ' ', 'id': ' ', 'change': True}
        )
        return f'id_{doc_id}'
    
    ###############################################################################
    # Common interface: returns a list of top-k document texts
    def search(self, query: str, n_results: int = 5, rich: bool = False):
        """If rich=True, returns a list of dicts with ids/metadatas/documents."""
        if rich:
            return self.search_rich(query, n_results=n_results)
        return self.retrieve(query, n_results)

    # Revised version of VectorStore.search_rich
    def search_rich(self, query: str, n_results: int = 5):
        """
        Returns: List[Dict] each = {"doc_id", "internal_id", "title", "text", "score"}
            - doc_id: Original document ID aligned 1:1 with qrels corpus-id (from metadata)
            - internal_id: Internal Chroma ID (e.g., id_123)
        """
        text_emb = self.get_embedding(query)
        res = self.collection.query(
            query_embeddings=[text_emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        ids   = res.get("ids", [[]])[0]                # internal id (id_123)
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

        out = []
        for i, txt in enumerate(docs):
            md = metas[i] if isinstance(metas, list) and i < len(metas) and isinstance(metas[i], dict) else {}
            internal_id = ids[i] if isinstance(ids, list) and i < len(ids) else None
            corpus_id   = md.get("id")  # ★ Original BEIR corpus-id matched with qrels
            title = md.get("title", "")
            score = None
            if isinstance(dists, list) and i < len(dists) and dists[i] is not None:
                try:
                    score = 1.0 - float(dists[i])  # cosine distance → similarity
                except Exception:
                    score = None
            out.append({"doc_id": corpus_id, "internal_id": internal_id, "title": title, "text": txt, "score": score})
        return out
    
    def search_ids_for_eval(self, query: str, n_results: int = 5):
        """For nDCG evaluation: returns only BEIR corpus-ids matching qrels."""
        rich = self.search_rich(query, n_results=n_results)
        return [r["doc_id"] for r in rich if r.get("doc_id")]

    def retrieve(self, query: str, n_results: int = 5):
        """Legacy compatibility: returns only a list of top-k document texts."""
        text_emb = self.get_embedding(query)
        res = self.collection.query(
            query_embeddings=[text_emb],  # 2D shape!
            n_results=n_results,
            include=["documents"]
        )
        return res.get("documents", [[]])[0]
    

def check_collection(collection_name):
    chroma_client = chromadb.PersistentClient(path=ChromadbPath, settings=Settings())
    collections = chroma_client.list_collections()
    exists = any(col.name == collection_name for col in collections)
    count = 0
    if exists:
        count = chroma_client.get_collection(name=collection_name).count()
        print(f"Collection exists: {collection_name}, Total items: {count}")
    else:
        print(f"Collection does not exist: {collection_name}")
    return exists, count

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument("--eval_dataset", type=str, default="msmarco")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--score_function", type=str, default="cosine")
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'

    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    collection_name = f"{args.eval_dataset}_{args.eval_model_code}_{args.score_function}"
    exists, count = check_collection(collection_name)

    store = VectorStore(model, tokenizer, get_emb, corpus, device, collection_name, use_local=exists and len(corpus) == count)

    if not (exists and len(corpus) == count):
        store.populate_vectors()

    store.update_context('id_3', 'ok,ok,ok')
    print(store.get_id('id_3'))
    store.clean_collect()
    print(store.get_id('id_3'))



# total 3633 
# python rag/vectorstore.py --eval_dataset 'nfcorpus' --gpu_id 3
# total 171332 
# python rag/vectorstore.py --eval_dataset 'trec-covid'
# total 2681468 1484090
# python rag/vectorstore.py --eval_dataset 'hotpotqa' --eval_model_code "contriever-msmarco"  --score_function 'cosine'
# total 8841823 1484090
# python rag/vectorstore.py --eval_dataset 'msmarco' --gpu_id 0
# total 5233329
# python rag/vectorstore.py --eval_dataset  'hotpotqa'
