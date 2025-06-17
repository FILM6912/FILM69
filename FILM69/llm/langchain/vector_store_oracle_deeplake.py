import os
import dotenv
import oracledb
import deeplake
import json

from typing import (
    Callable, List, Union, Literal, Optional, Any, Iterable
)
from collections.abc import Collection, Iterator, Sequence
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_deeplake.vectorstores import DeeplakeVectorStore

dotenv.load_dotenv()



"""
retriever = db.db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 20
"""

class VectorStoreOracleDeeplake(VectorStore):
    def __init__(
        self,
        vectorstore: Literal["Deeplake", "Oracle"],
        embedding_function: Union[Callable[[str], List[float]], Embeddings],
    ):
        """
        สร้างอินสแตนซ์ของ VectorStoreOracleDeeplake โดยสามารถเลือกใช้ vectorstore ได้ระหว่าง "Deeplake" หรือ "Oracle"
        พร้อมกำหนด embedding_function สำหรับแปลงข้อความเป็นเวกเตอร์

        พารามิเตอร์:
        - vectorstore (Literal["Deeplake", "Oracle"]): ระบุว่าจะใช้ระบบจัดเก็บเวกเตอร์แบบใด
        - embedding_function (Callable หรือ Embeddings): ฟังก์ชันหรืออ็อบเจ็กต์ที่ใช้แปลงข้อความเป็นเวกเตอร์

        การตั้งค่าที่ต้องกำหนดไว้ในไฟล์ .env:
        - oracledb_user="user"                         # ชื่อผู้ใช้ฐานข้อมูล Oracle
        - oracledb_password="password"                 # รหัสผ่านของผู้ใช้
        - oracledb_dsn="localhost:1521/FREE"           # ข้อมูลการเชื่อมต่อ Oracle (host:port/service_name)
        - oracledb_table_name="VECTORDB_{version}"     # ชื่อตารางใน Oracle ที่ใช้เก็บเวกเตอร์
        - deeplake_dataset_path="file://data_{version}"# พาธไปยัง Dataset ของ Deeplake
        """
        self.vectorstore = vectorstore
        self.embedding_function = embedding_function
        self.db = None
        self.version = None

    ############################# Load/Reload #############################

    def load(self, version: int):
        self.version = version
        self.db, self.database_table_or_path = self._load(version)

    def reload(self):
        self.load(self.version)

    def _load(self, version: int):
        try:
            if self.vectorstore == "Deeplake":
                dataset_path = os.getenv("deeplake_dataset_path").replace("{version}", f"{version}")
                db = DeeplakeVectorStore(
                    dataset_path=dataset_path,
                    embedding_function=self.embedding_function
                )
                return db, dataset_path

            elif self.vectorstore == "Oracle":
                table_name = os.getenv("oracledb_table_name").replace("{version}", f"{version}")
                connection = oracledb.connect(
                    user=os.getenv("oracledb_user"),
                    password=os.getenv("oracledb_password"),
                    dsn=os.getenv("oracledb_dsn")
                )
                self.cursor = connection.cursor()
                db = OracleVS(
                    client=connection,
                    table_name=table_name,
                    embedding_function=self.embedding_function,
                    distance_strategy=DistanceStrategy.COSINE
                )
                return db, table_name

        except Exception:
            return {"status": "Connection failed!"}

    ########################## Basic VectorStore Methods ##########################

    def from_texts(
        self,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        self.db.from_texts(texts, embedding, metadatas, ids=ids, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        return self.db.add_texts(texts, metadatas, ids=ids, **kwargs)

    def add_documents(self, documents, **kwargs):
        return self.db.add_documents(documents, **kwargs)

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        if self.vectorstore == "Deeplake":
            return self.db.delete(ids=ids, **kwargs)

        elif self.vectorstore == "Oracle":
            byte_ids = [bytes.fromhex(hex_id) for hex_id in ids]
            self.cursor.executemany(
                f"DELETE FROM {self.database_table_or_path.replace('{version}', f'{self.version}')} WHERE ID = :id",
                [{"id": b} for b in byte_ids]
            )
            self.db.client.commit()
            return True

    ############################# Search Methods #############################

    def search(self, query, search_type, **kwargs):
        return self.db.search(query, search_type, **kwargs)

    def similarity_search(self, query, k=4, **kwargs):
        return self.db.similarity_search(query, k, **kwargs)

    def similarity_search_with_score(self, *args, **kwargs):
        return self.db.similarity_search_with_score(*args, **kwargs)

    def similarity_search_by_vector(self, embedding, k=4, **kwargs):
        return self.db.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        return self.db.similarity_search_with_relevance_scores(query, k, **kwargs)

    def get_by_ids(self, ids, **kwargs):
        return self.db.get_by_ids(ids, **kwargs)

    ############################# Update Methods #############################

    def update(self, ids: list[str], text: list[str]) -> str:
        try:
            self.delete(ids=ids)
            self.db.add_texts(text, ids=ids)
            return "Success"
        except Exception as e:
            return str(e)

    def used_version(self) -> int:
        return self.version

    ########################### Version-Aware Methods ###########################

    def add_texts_by_version(self, version, texts: list[str]) -> str:
        try:
            db, _ = self._load(version)
            out = db.add_texts(texts)
            self.reload()
            return out
        except Exception as e:
            return str(e)

    def add_documents_by_version(self, version, documents) -> str:
        try:
            db, _ = self._load(version)
            db.add_documents(documents)
            self.reload()
            return "Success"
        except Exception as e:
            return str(e)

    def update_text_by_version(self, version, ids: list[str], text: list[str]) -> str:
        try:
            db, _ = self._load(version)
            db.delete(version=version, ids=ids)
            db.add_texts(text, ids=ids)
            self.reload()
            return "Success"
        except Exception as e:
            return str(e)

    def get_by_version(self, version=None, **kwargs) -> list[dict]:
        if self.vectorstore == "Deeplake":
            db, _ = self._load(version)
            dataset = db.dataset
            rows = []
            for i in range(min(4, len(dataset))):
                sample = {
                    "ids": dataset["ids"][i],
                    "metadata": [{k: v} for k, v in dataset["metadata"][i].items()],
                    "documents": dataset["documents"][i],
                }
                rows.append(sample)
            return rows

        elif self.vectorstore == "Oracle":
            self.cursor.execute(
                f"SELECT ID, TEXT, METADATA FROM {self.database_table_or_path.replace('{version}', f'{version}')}"
            )
            rows = self.cursor.fetchall()
            result = []
            for id_, text, meta in rows:
                id_hex = id_.hex() if hasattr(id_, "hex") else id_
                doc = text.read() if hasattr(text, "read") else text
                meta_str = meta.read() if hasattr(meta, "read") else meta
                try:
                    meta_json = json.loads(meta_str) if meta_str else {}
                except:
                    meta_json = meta_str
                result.append({
                    "ids": id_hex,
                    "metadata": meta_json,
                    "documents": doc
                })
            return result

    def delete_documents_by_version(self, version, ids: Optional[list[str]] = None, **kwargs) -> str:
        try:
            if self.vectorstore == "Deeplake":
                db, _ = self._load(version)
                out = db.delete(ids=ids, **kwargs)

            elif self.vectorstore == "Oracle":
                byte_ids = [bytes.fromhex(hex_id) for hex_id in ids]
                self.cursor.executemany(
                    f"DELETE FROM {self.database_table_or_path.replace('{version}', f'{version}')} WHERE ID = :id",
                    [{"id": b} for b in byte_ids]
                )
                self.db.client.commit()
                out = True

            self.reload()
            return out

        except Exception as e:
            return str(e)

    def create_new_version(self, new_version, use_data_version=None) -> dict:
        try:
            if self.vectorstore == "Deeplake":
                deeplake.copy(
                    src=self.database_table_or_path.replace('{version}', f'{use_data_version}'),
                    dst=self.database_table_or_path.replace('{version}', f'{new_version}'),
                )
                
            elif self.vectorstore == "Oracle" and use_data_version is None:
                self.load(new_version)
                
            elif self.vectorstore == "Oracle" and use_data_version is not None:
                self.cursor.execute(f"""
                    CREATE TABLE {self.database_table_or_path.replace('{version}', f'{new_version}')} AS
                    SELECT * FROM {self.database_table_or_path.replace('{version}', f'{use_data_version}')}
                """)
                self.load(new_version)
                
            

            return {"status": "Success", "version": new_version}

        except Exception as e:
            return {"status": str(e)}

    def delete_version(self, version) -> dict:
        try:
            if self.vectorstore == "Deeplake":
                deeplake.delete(self.database_table_or_path.replace('{version}', f'{version}'))

            elif self.vectorstore == "Oracle":
                self.cursor.execute(f"DROP TABLE {self.database_table_or_path.replace('{version}', f'{version}')}")

            return {"status": "Success", "version": version}
        except Exception as e:
            return {"status": str(e)}
        
    def get_versions(self) -> list:
        if self.vectorstore == "Deeplake":
            return None
        elif self.vectorstore == "Oracle":
            self.cursor.execute(f"""
                SELECT table_name 
                FROM user_tables 
                WHERE table_name LIKE '%{self.database_table_or_path.replace(str(self.version), f'')}%'
            """)
            return [row[0].split("_")[-1] for row in self.cursor.fetchall()]
            
