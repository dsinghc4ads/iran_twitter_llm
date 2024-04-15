import os
import chromadb
from dotenv import load_dotenv
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import (
    SimpleKeywordTableIndex,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.core.storage.docstore import SimpleDocumentStore
from fastapi import FastAPI
from openai import BadRequestError
from pydantic import BaseModel
import uvicorn

from utils.csv_reader import PandasCSVReader
from utils.retrievers import HybridRetriever
import logging

logging.basicConfig(level=logging.INFO)

# Load variables from .env file
load_dotenv(".env")
openai_model = os.getenv("OPENAI_MODEL")


def generate_docs():
    logging.info("Loading documents")
    data_path = os.getenv("DATA_PATH")
    documents = PandasCSVReader(
        concat_rows=False, exclude_metadata_columns=["text"]
    ).load_data(data_path)
    for doc in documents:
        doc.metadata['doc_id'] = doc.id_
    return documents


def process_documents(documents, batch_size=10):
    logging.info("Processing documents")
    persist_dir = os.getenv("PERSIST_DIR")
    doc_store_persist_path = os.path.join(persist_dir, "docstore.json")
    logging.info(f"Persisting to {persist_dir}")
    log_file_path = "error_log.txt"
    logging.basicConfig(
        filename=log_file_path,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if not os.listdir(persist_dir):
        logging.info("Creating new collection")
        chroma_client = chromadb.PersistentClient()
        chroma_collection = chroma_client.create_collection("quickstart")
        logging.info(f"Collection created: {chroma_collection}")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        logging.info(f"Vector store created: {vector_store}")
        # Configure Ingest Cache
        ingest_cache = IngestionCache(
            cache=RedisCache.from_host_and_port(host="redis", port=6379),
            collection="llama_cache",
        )
        logging.info(f"Ingest cache created: {ingest_cache}")

        # Step 3 split documents into Nodes
        logging.info("Ingesting new collection")
        doc_store = SimpleDocumentStore()
        pipeline = IngestionPipeline(
            transformations=[
                SemanticSplitterNodeParser(embed_model=OpenAIEmbedding()),
                TitleExtractor(),
                OpenAIEmbedding(),
            ],
            vector_store=vector_store,
            cache=ingest_cache,
            docstore=doc_store,
        )

        # Process documents in batches
        def process_batch(doc_batch):
            try:
                pipeline.run(documents=doc_batch)
            except BadRequestError as e:
                if len(doc_batch) == 1:
                    logging.error(f"Error processing document: {e}")
                else:
                    mid = len(doc_batch) // 2
                    process_batch(doc_batch[:mid])
                    process_batch(doc_batch[mid:])

        total_documents = len(documents)
        for i in range(0, total_documents, batch_size):
            batch = documents[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{(total_documents + batch_size - 1) // batch_size}")
            process_batch(batch)

        # Persist the document store
        doc_store.persist(doc_store_persist_path)

    else:
        logging.info("Loading existing collection")
        chroma_client = chromadb.PersistentClient()
        chroma_collection = chroma_client.get_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        doc_store = SimpleDocumentStore.from_persist_dir(persist_dir)

    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    keyword_index = SimpleKeywordTableIndex(nodes=list(doc_store.docs.values()))

    return vector_index, keyword_index


def create_query_engines(vector_index, keyword_index):
    """
    Creates query engines for vector, keyword, and hybrid retrievers
    """
    logging.info("Creating query engine")
    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=5,
    )
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
    hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever)
    response_synthesizer = get_response_synthesizer()
    hybrid_query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
    )
    vector_query_engine = RetrieverQueryEngine(
        retriever=vector_retriever,
        response_synthesizer=response_synthesizer,
    )
    keyword_query_engine = RetrieverQueryEngine(
        retriever=keyword_retriever,
        response_synthesizer=response_synthesizer,
    )
    return hybrid_query_engine, vector_query_engine, keyword_query_engine


class QueryRequest(BaseModel):
    question: str


app = FastAPI()


@app.post("/query/")
async def query(request: QueryRequest):
    # check if "mode" is in request
    if "mode" in request:
        mode = request.mode
    else:
        mode = "hybrid"
    response = query_engine_map[mode].query(request.question)
    formatted_response = {
        "response": response,
        "relevant_sources": response.get_formatted_sources(length=10000),
    }
    return formatted_response


if __name__ == "__main__":
    try:
        documents = generate_docs()
        logging.info("Creating index")
        vector_index, keyword_index = process_documents(documents)
        logging.info("Indexing complete, creating query engines")
        hybrid_query_engine, vector_query_engine, keyword_query_engine = (
            create_query_engines(vector_index, keyword_index)
        )
        logging.info("Query engines created")
        query_engine_map = {
            "hybrid": hybrid_query_engine,
            "vector": vector_query_engine,
            "keyword": keyword_query_engine,
        }

        uvicorn.run(app, host="0.0.0.0", port=8080)
    except Exception as e:
        logging.info(e)
        raise e
