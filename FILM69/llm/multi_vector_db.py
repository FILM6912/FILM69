import chromadb
import uuid
from collections import defaultdict
from em import EmbeddingModel
from chromadb.config import Settings
import torch
torch.set_float32_matmul_precision('high')

class MultiVectorDB:
    def __init__(self,db_name="embedding_DB"):
        self.client = chromadb.PersistentClient(db_name)
        self.collection = self.client.get_or_create_collection(
            name="vector",
            embedding_function=None,
            metadata={
                "hnsw:M": 500,
                "hnsw:construction_ef": 600,
                "hnsw:space": "l2"
            }
        )
        
        self.em = EmbeddingModel()

    def add(self, documents: list[str], ids: list[str] = None, metadatas: list[dict] = None):
        id = ids
        meta = metadatas
        embedding = self.em(documents)
        ids, embeddings, metadatas, docs = [], [], [], []

        if meta is None:
            meta = [None] * len(documents)

        for i, (doc, text_doc, meta_data) in enumerate(zip(embedding, documents, meta)):
            random_uuid = id[i] if id is not None else uuid.uuid4()
            index = 0
            text_split = self._split_text(text_doc, len(doc))
            for j, (embed, text) in enumerate(zip(doc, text_split)):
                index += 1
                ids.append(f"{random_uuid}_{index}")
                embeddings.append(embed)
                try:
                    metadatas.append({"doc_id": f"{random_uuid}", **meta_data})
                except:
                    metadatas.append({"doc_id": f"{random_uuid}"})
                docs.append(text)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=docs
        )

    def query(self, query_text: str, n_results=10, max=1000):
        try:
            embedding = self.em(query_text)[0]
            results = self.collection.query(query_embeddings=embedding, n_results=max)
            distances, ids = results["distances"][0], results["ids"][0]

            scores = {}
            for doc_id, dist in zip(ids, distances):
                prefix = doc_id.split("_")[0]
                scores[prefix] = min(dist, scores.get(prefix, float("inf")))

            target_ids = [f"{prefix}_1" for prefix in scores]
            res = self.collection.get(ids=target_ids)

            distance_map = {f"{k}_1": v for k, v in scores.items()}
            combined = [{
                "id": doc_id,
                "distance": distance_map.get(doc_id),
                "document": doc,
                "metadata": meta
            } for doc_id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]

            combined_doc = self.collection.get(where={"doc_id": {"$in": list(scores.keys())}})
            combined_doc = dict(self._merge_documents_by_doc_id(combined_doc, metadata=False).items())

            res = sorted(combined, key=lambda x: x["distance"])[:n_results]

            all_docs = []
            for i in range(len(res)):
                doc_id_prefix = res[i]["id"].split("_")[0]
                res[i]["document"] = combined_doc[doc_id_prefix]
                all_docs.append(res[i]["document"])

            score_retrieval = self.em.score_retrieval([embedding], self.em(all_docs))[0]

            for i in range(len(res)):
                res[i]["distance"] = score_retrieval[i]
                res[i]["id"] = res[i]["id"].split("_")[0]

            res = sorted(res, key=lambda x: x["distance"], reverse=True)[:n_results]
            return res
        except:
            return []


    def _split_text(self, text, num_chunks):
        avg_chunk_length = len(text) // num_chunks
        remainder = len(text) % num_chunks

        chunks, start = [], 0
        for i in range(num_chunks):
            end = start + avg_chunk_length + (1 if i < remainder else 0)
            chunks.append(text[start:end])
            start = end
        return chunks

    def _merge_documents_by_doc_id(self, results, metadata=True):
        merged_docs = defaultdict(list)
        merged_metas = {}

        for doc, meta in zip(results["documents"], results["metadatas"]):
            doc_id = meta["doc_id"]
            merged_docs[doc_id].append(doc)
            if doc_id not in merged_metas:
                merged_metas[doc_id] = meta

        if metadata:
            return [
                {
                    "id": doc_id,
                    "text": ''.join(chunks),
                    "metadata": merged_metas[doc_id]
                }
                for doc_id, chunks in merged_docs.items()
            ]
        else:
            return {doc_id: ''.join(chunks) for doc_id, chunks in merged_docs.items()}

    def delete(self, ids: list[str]):
        self.collection.delete(where={"doc_id": {"$in": ids}})

    def update(self, ids: list[str], documents: list[str]):
        self.delete(ids)
        self.add(documents, ids)

    def get(self, ids: list[str] = None, **kwargs):
        return self._merge_documents_by_doc_id(self.collection.get(ids=ids, **kwargs))


if __name__ == "__main__":
    db = MultiVectorDB()
    data = ["สวัสดี", "คุณคือ"]
    db.add(documents=data)
    res = db.query("Communication Protocol")

    db.update(["4a8af005-4f6e-4ff1-a170-b7ced4eb2a47"], ["5555"])
    db.delete(["e718654c-79fa-49ac-afde-89d012eabcac"])
