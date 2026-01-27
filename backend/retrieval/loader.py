import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests
from langchain_core.documents import Document

from retrieval.constants import DOCS_SOURCES

logger = logging.getLogger("langgraph_agent")


class DocumentLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.metadata_file = self.raw_dir / "metadata.json"

    def download_docs(self, force: bool = False) -> Dict:
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "sources": {},
        }

        for name, source in DOCS_SOURCES.items():
            file_path = self.raw_dir / f"{name}.txt"

            if file_path.exists() and not force:
                logger.info(f"[LOADER] Skipping {name} (already exists)")
                if self.metadata_file.exists():
                    existing = json.loads(self.metadata_file.read_text())
                    if name in existing.get("sources", {}):
                        metadata["sources"][name] = existing["sources"][name]
                continue

            try:
                logger.info(f"[LOADER] Downloading {name} from {source['url']}")
                response = requests.get(source["url"], timeout=60)
                response.raise_for_status()

                content = response.text
                file_path.write_text(content, encoding="utf-8")

                metadata["sources"][name] = {
                    "url": source["url"],
                    "description": source["description"],
                    "priority": source["priority"],
                    "size_bytes": len(content),
                    "size_chars": len(content),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }

                logger.info(f"[LOADER] Downloaded {name}: {len(content):,} chars")

            except requests.RequestException as e:
                logger.error(f"[LOADER] Failed to download {name}: {e}")
                metadata["sources"][name] = {
                    "url": source["url"],
                    "error": str(e),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }

        self.metadata_file.write_text(json.dumps(metadata, indent=2))
        logger.info(f"[LOADER] Metadata saved to {self.metadata_file}")

        return metadata

    def load_documents(self) -> List[Document]:
        documents = []

        for name, source in DOCS_SOURCES.items():
            file_path = self.raw_dir / f"{name}.txt"

            if not file_path.exists():
                logger.warning(f"[LOADER] File not found: {file_path}")
                continue

            content = file_path.read_text(encoding="utf-8")

            doc = Document(
                page_content=content,
                metadata={
                    "source": name,
                    "url": source["url"],
                    "description": source["description"],
                    "priority": source["priority"],
                    "file_path": str(file_path),
                },
            )
            documents.append(doc)
            logger.info(f"[LOADER] Loaded {name}: {len(content):,} chars")

        return documents

    def get_metadata(self) -> Optional[Dict]:
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return None

    def get_doc_age_hours(self) -> Optional[float]:
        metadata = self.get_metadata()
        if not metadata or "downloaded_at" not in metadata:
            return None

        downloaded_at = datetime.fromisoformat(metadata["downloaded_at"])
        now = datetime.now(timezone.utc)

        if downloaded_at.tzinfo is None:
            downloaded_at = downloaded_at.replace(tzinfo=timezone.utc)

        age = now - downloaded_at
        return age.total_seconds() / 3600
