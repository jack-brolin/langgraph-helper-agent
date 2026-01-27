import logging
import time
from pathlib import Path
from typing import List, Optional, Callable

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from retrieval.constants import (
    PARENT_COLLECTION,
    CHILD_COLLECTION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_DELAY,
    MAX_RETRIES,
)
from retrieval.loader import DocumentLoader
from retrieval.chunker import DocumentChunker

logger = logging.getLogger("langgraph_agent")


class VectorStoreManager:
    def __init__(
        self,
        data_dir: Path,
        google_api_key: str,
        embedding_model: str = "models/text-embedding-004",
        chroma_host: Optional[str] = None,
        chroma_port: int = 8000,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_delay: float = DEFAULT_BATCH_DELAY,
    ):
        self.data_dir = Path(data_dir)
        self.chroma_dir = self.data_dir / "chroma"
        self.google_api_key = google_api_key
        self.embedding_model = embedding_model
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.batch_size = batch_size
        self.batch_delay = batch_delay

        self.use_server = chroma_host is not None and chroma_host != ""

        if self.use_server:
            logger.info(f"[STORE] Using Chroma server at {chroma_host}:{chroma_port}")
        else:
            logger.info(f"[STORE] Using local Chroma storage at {self.chroma_dir}")

        self.loader = DocumentLoader(self.data_dir)
        self.chunker = DocumentChunker()

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key,
        )

        self._client: Optional[chromadb.ClientAPI] = None
        self._child_vectorstore: Optional[Chroma] = None
        self._parent_vectorstore: Optional[Chroma] = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Get or create the Chroma client."""
        if self._client is None:
            if self.use_server:
                # Connect to Chroma server
                self._client = chromadb.HttpClient(
                    host=self.chroma_host,
                    port=self.chroma_port,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                    ),
                )
                logger.info(f"[STORE] Connected to Chroma server at {self.chroma_host}:{self.chroma_port}")
            else:
                # Use persistent local storage
                self.chroma_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.chroma_dir),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                    ),
                )
                logger.info(f"[STORE] Using local Chroma storage at {self.chroma_dir}")

        return self._client

    @property
    def child_vectorstore(self) -> Chroma:
        """Get or create the child vector store (for search)."""
        if self._child_vectorstore is None:
            client = self._get_client()
            self._child_vectorstore = Chroma(
                client=client,
                embedding_function=self.embeddings,
                collection_name=CHILD_COLLECTION,
            )
        return self._child_vectorstore
    
    @property
    def parent_vectorstore(self) -> Chroma:
        """Get or create the parent vector store (for retrieval)."""
        if self._parent_vectorstore is None:
            client = self._get_client()
            self._parent_vectorstore = Chroma(
                client=client,
                embedding_function=self.embeddings,
                collection_name=PARENT_COLLECTION,
            )
        return self._parent_vectorstore

    def index_exists(self) -> bool:
        """Check if the child vector store index exists and has documents."""
        try:
            client = self._get_client()
            
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            if CHILD_COLLECTION not in collection_names:
                return False

            collection = client.get_collection(CHILD_COLLECTION)
            count = collection.count()
            return count > 0

        except Exception as e:
            logger.warning(f"[STORE] Error checking index: {e}")
            return False

    def get_index_stats(self) -> dict:
        """Get statistics about the parent-child vector store index."""
        try:
            if not self.index_exists():
                return {"exists": False, "child_count": 0, "parent_count": 0}

            client = self._get_client()
            child_collection = client.get_collection(CHILD_COLLECTION)
            child_count = child_collection.count()
            
            parent_count = 0
            collections = client.list_collections()
            if PARENT_COLLECTION in [c.name for c in collections]:
                parent_collection = client.get_collection(PARENT_COLLECTION)
                parent_count = parent_collection.count()

            doc_metadata = self.loader.get_metadata()

            stats = {
                "exists": True,
                "child_count": child_count,
                "parent_count": parent_count,
                "document_count": child_count,  # For backward compat
                "chunking_strategy": "parent-child",
                "doc_metadata": doc_metadata,
                "doc_age_hours": self.loader.get_doc_age_hours(),
                "storage_mode": "server" if self.use_server else "local",
            }

            if self.use_server:
                stats["chroma_server"] = f"{self.chroma_host}:{self.chroma_port}"
            else:
                stats["chroma_dir"] = str(self.chroma_dir)

            return stats

        except Exception as e:
            logger.error(f"[STORE] Error getting stats: {e}")
            return {"exists": False, "error": str(e)}

    def _add_documents_batch(
        self,
        documents: List[Document],
        vectorstore: Chroma,
        batch_num: int,
        total_batches: int,
        doc_type: str = "document",
    ) -> int:
        """
        Add a batch of documents to the vector store with retry logic.

        Args:
            documents: List of documents to add
            vectorstore: The Chroma vectorstore to add to
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches
            doc_type: Type of documents (for logging)

        Returns:
            Number of documents successfully added
        """
        for attempt in range(MAX_RETRIES):
            try:
                vectorstore.add_documents(documents)
                
                logger.info(
                    f"[STORE] Batch {batch_num}/{total_batches}: "
                    f"Added {len(documents)} {doc_type}s"
                )
                return len(documents)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(
                        f"[STORE] Batch {batch_num} failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"[STORE] Batch {batch_num} failed after {MAX_RETRIES} attempts: {e}"
                    )
                    raise

        return 0

    def build_index_batched(
        self,
        force_download: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        def log_progress(current: int, total: int, message: str):
            """Log progress and call callback if provided."""
            logger.info(f"[STORE] Progress: {current}/{total} - {message}")
            if progress_callback:
                progress_callback(current, total, message)

        logger.info("[STORE] Starting batched index build...")
        start_time = time.time()

        log_progress(0, 5, "Downloading documentation...")
        download_meta = self.loader.download_docs(force=force_download)
        logger.info(f"[STORE] Downloaded {len(download_meta['sources'])} sources")

        log_progress(1, 5, "Loading documents...")
        documents = self.loader.load_documents()
        if not documents:
            raise ValueError("No documents loaded. Download may have failed.")

        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(f"[STORE] Loaded {len(documents)} documents ({total_chars:,} total characters)")

        log_progress(2, 5, "Chunking documents with parent-child strategy...")
        parents, children = self.chunker.chunk_documents(documents)
        logger.info(f"[STORE] Created {len(parents)} parents, {len(children)} children")

        log_progress(3, 5, "Preparing vector stores...")
        try:
            client = self._get_client()
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            # Delete old collections if they exist
            for coll_name in [CHILD_COLLECTION, PARENT_COLLECTION]:
                if coll_name in collection_names:
                    client.delete_collection(coll_name)
                    logger.info(f"[STORE] Deleted existing collection: {coll_name}")
            
            # Reset vectorstore references
            self._child_vectorstore = None
            self._parent_vectorstore = None
            
        except Exception as e:
            logger.warning(f"[STORE] Could not delete old collections: {e}")

        # Index children (for search)
        log_progress(4, 6, f"Indexing {len(children)} children in batches of {self.batch_size}...")
        
        total_child_batches = (len(children) + self.batch_size - 1) // self.batch_size
        total_children_indexed = 0
        failed_child_batches = []

        for batch_num in range(total_child_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(children))
            batch = children[start_idx:end_idx]

            try:
                indexed = self._add_documents_batch(
                    documents=batch,
                    vectorstore=self.child_vectorstore,
                    batch_num=batch_num + 1,
                    total_batches=total_child_batches,
                    doc_type="child",
                )
                total_children_indexed += indexed

            except Exception as e:
                logger.error(f"[STORE] Failed to index child batch {batch_num + 1}: {e}")
                failed_child_batches.append(batch_num + 1)

            if batch_num < total_child_batches - 1:
                time.sleep(self.batch_delay)

        # Index parents (for retrieval)
        log_progress(5, 6, f"Indexing {len(parents)} parents in batches of {self.batch_size}...")
        
        total_parent_batches = (len(parents) + self.batch_size - 1) // self.batch_size
        total_parents_indexed = 0
        failed_parent_batches = []

        for batch_num in range(total_parent_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(parents))
            batch = parents[start_idx:end_idx]

            try:
                indexed = self._add_documents_batch(
                    documents=batch,
                    vectorstore=self.parent_vectorstore,
                    batch_num=batch_num + 1,
                    total_batches=total_parent_batches,
                    doc_type="parent",
                )
                total_parents_indexed += indexed

            except Exception as e:
                logger.error(f"[STORE] Failed to index parent batch {batch_num + 1}: {e}")
                failed_parent_batches.append(batch_num + 1)

            if batch_num < total_parent_batches - 1:
                time.sleep(self.batch_delay)

        elapsed_time = time.time() - start_time
        log_progress(6, 6, f"Indexing complete in {elapsed_time:.1f}s")

        failed_batches = failed_child_batches + failed_parent_batches
        result = {
            "success": len(failed_batches) == 0,
            "documents_loaded": len(documents),
            "parents_created": len(parents),
            "children_created": len(children),
            "parents_indexed": total_parents_indexed,
            "children_indexed": total_children_indexed,
            "chunks_created": len(children),  # For backward compat
            "chunks_indexed": total_children_indexed,  # For backward compat
            "total_batches": total_child_batches + total_parent_batches,
            "failed_batches": failed_batches,
            "elapsed_seconds": round(elapsed_time, 1),
            "download_metadata": download_meta,
            "chunking_strategy": "parent-child",
        }

        if failed_batches:
            logger.warning(f"[STORE] {len(failed_batches)} batches failed: {failed_batches}")
        else:
            logger.info(
                f"[STORE] Index built successfully: {total_parents_indexed} parents, "
                f"{total_children_indexed} children in {elapsed_time:.1f}s"
            )

        return result

    def build_index(self, force_download: bool = False) -> dict:
        return self.build_index_batched(force_download=force_download)

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> List[tuple]:
        """
        Search using parent-child retrieval.
        
        Searches in child collection (precise matches), 
        but returns parent documents (full context).
        """
        if not self.index_exists():
            logger.warning("[STORE] Index does not exist, returning empty results")
            return []

        try:
            # Search in children (small, precise matches)
            child_results = self.child_vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict,
            )

            if not child_results:
                return []

            # Extract unique parent IDs from matched children
            parent_ids = list(set(doc.metadata.get("parent_id") for doc, _ in child_results))
            
            # Retrieve parent documents
            parent_docs = []
            for parent_id in parent_ids:
                parent_result = self.parent_vectorstore.get(
                    where={"parent_id": parent_id}
                )
                if parent_result and parent_result.get("documents"):
                    # Reconstruct Document with metadata
                    for i, doc_content in enumerate(parent_result["documents"]):
                        metadata = parent_result["metadatas"][i] if parent_result.get("metadatas") else {}
                        from langchain_core.documents import Document
                        parent_doc = Document(page_content=doc_content, metadata=metadata)
                        # Use the child's score since parent isn't scored
                        matching_child_score = next(
                            (score for doc, score in child_results if doc.metadata.get("parent_id") == parent_id),
                            0.5
                        )
                        parent_docs.append((parent_doc, matching_child_score))

            logger.info(
                f"[STORE] Search for '{query[:50]}...' "
                f"matched {len(child_results)} children, returning {len(parent_docs)} parents"
            )
            return parent_docs[:k]  # Return top k parents

        except Exception as e:
            logger.error(f"[STORE] Search error: {e}")
            return []


def create_store_manager(
    data_dir: Path,
    google_api_key: str,
    chroma_host: Optional[str] = None,
    chroma_port: int = 8000,
    **kwargs,
) -> VectorStoreManager:
    return VectorStoreManager(
        data_dir=data_dir,
        google_api_key=google_api_key,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        **kwargs,
    )
