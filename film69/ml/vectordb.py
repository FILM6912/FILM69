from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Optional, Union
from chromadb.api.types import (
    URI,
    CollectionMetadata,
    Embedding,
    Include,
    Metadata,
    Document,
    Image,
    Where,
    IDs,
    GetResult,
    QueryResult,
    ID,
    OneOrMany,
    WhereDocument,
    EmbeddingFunction
)

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self,embedding_name):
        self.model = SentenceTransformer(embedding_name)
    def __call__(self, inputs):
        return self.model.encode(inputs).tolist()

class VectorDB:
    def __init__(self,path="database", collection_name="data", embedding_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        client = chromadb.PersistentClient(path=path)
        self.db = client.get_or_create_collection(collection_name,embedding_function=CustomEmbeddingFunction(embedding_name))
        print("Loaded successfully")
    
    def add_or_update(self, 
        ids: OneOrMany[ID],
        embeddings: Optional[Union[OneOrMany[Embedding],OneOrMany[np.ndarray],]] = None,
        metadatas: Optional[OneOrMany[Metadata]] = None,
        documents: Optional[OneOrMany[Document]] = None,
        images: Optional[OneOrMany[Image]] = None,
        uris: Optional[OneOrMany[URI]] = None,):
        self.db.upsert(ids=ids,embeddings=embeddings,metadatas=metadatas,documents=documents,images=images,uris=uris)
    
    def query(self,
        query_embeddings: Optional[Union[OneOrMany[Embedding],OneOrMany[np.ndarray],]] = None,
        query_texts: Optional[OneOrMany[Document]] = None,
        query_images: Optional[OneOrMany[Image]] = None,
        query_uris: Optional[OneOrMany[URI]] = None,
        n_results: int = 10,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,
        include: Include = ["metadatas", "documents", "distances"],
        on_dict:bool=False,
        metadata_columns:list[str]=None):
        result = self.db.query(query_embeddings=query_embeddings,query_texts=query_texts,query_images=query_images,query_uris=query_uris,n_results=n_results,where=where,where_document=where_document,include=include)
        data_df={
            "id":result["ids"][0],
            "document":result["documents"][0],
            "distance":result["distances"][0],
        }
        if metadata_columns != None:out=pd.concat([pd.DataFrame(data_df),pd.DataFrame(result["metadatas"][0])[metadata_columns]],axis=1)
        else:out=pd.concat([pd.DataFrame(data_df),pd.DataFrame(result["metadatas"][0])],axis=1)
        return out if not on_dict else out.to_dict()
    
    def delete(self,
        ids: Optional[IDs] = None,
        where: Optional[Where] = None,
        where_document: Optional[WhereDocument] = None,):
        self.db.delete(ids=ids,where=where,where_document=where_document)
    
    def get(self,  
        ids: Optional[OneOrMany[ID]] = None,
        where: Optional[Where] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        where_document: Optional[WhereDocument] = None,
        include: Include = ["metadatas", "documents"],
        on_dict:bool=False,metadata_columns:list[str]=None):
        result=self.db.get(ids=ids,where=where,limit=limit,offset=offset,where_document=where_document,include=include)
        data_df={"id":result["ids"],"document":result["documents"]}
        if metadata_columns != None:out=pd.concat([pd.DataFrame(data_df),pd.DataFrame(result["metadatas"])[metadata_columns]],axis=1)
        else:out=pd.concat([pd.DataFrame(data_df),pd.DataFrame(result["metadatas"])],axis=1)
        return out if not on_dict else out.to_dict()
    
        
    
    
        

    
    
        