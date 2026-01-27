import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from retrieval.store import VectorStoreManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_docs")


def main():
    parser = argparse.ArgumentParser(description="Index LangGraph/LangChain documentation")
    parser.add_argument("--chroma-host", type=str, required=True, help="Chroma server host")
    parser.add_argument("--chroma-port", type=int, required=True, help="Chroma server port")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-index")
    
    args = parser.parse_args()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("GOOGLE_API_KEY environment variable is required")
        sys.exit(1)

    # Resolve data directory: use env var if set, otherwise use project root's data/
    # This works regardless of where the script is called from
    data_dir = Path(os.getenv("DATA_DIR"))
    
    logger.info(f"Data directory: {data_dir.absolute()}")
    logger.info(f"Chroma server: {args.chroma_host}:{args.chroma_port}")

    store = VectorStoreManager(
        data_dir=data_dir,
        google_api_key=google_api_key,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
    )

    if store.index_exists() and not args.force:
        stats = store.get_index_stats()
        logger.info(f"Index already exists with {stats.get('document_count', 0)} documents")
        logger.info("Use --force to rebuild")
        return

    logger.info("Building documentation index...")
    
    try:
        result = store.build_index(force_download=args.force)

        logger.info("=" * 60)
        logger.info("Index Build Complete!")
        logger.info(f"  Documents loaded:  {result['documents_loaded']}")
        logger.info(f"  Chunks created:    {result['chunks_created']}")
        logger.info(f"  Chunks indexed:    {result.get('chunks_indexed', result['chunks_created'])}")
        logger.info(f"  Time elapsed:      {result.get('elapsed_seconds', 'N/A')}s")
        
        if result.get('failed_batches'):
            logger.warning(f"  Failed batches:    {len(result['failed_batches'])}")
        
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
