import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("langgraph_agent")


class DocumentChunker:
    """
    Creates parent-child chunk hierarchy for optimal RAG performance.
    
    Strategy:
    - Parent chunks: Large (2000 chars) for full LLM context
    - Child chunks: Small (500 chars) for precise retrieval
    - Children reference parents via parent_id in metadata
    
    This solves the RAG trade-off: small chunks find specific matches,
    but large parents provide complete context to the LLM.
    """
    
    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=[
                "\n```",
                "\n## ", 
                "\n### ",    
                "\n#### ",   
                "\n\n",      
                "\n",        
                ". ",        
                " ",         
                "",          
            ],
            keep_separator=True,
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=[
                "\n```",
                "\n### ",    
                "\n#### ",   
                "\n\n",      
                "\n",        
                ". ",        
                " ",         
                "",          
            ],
            keep_separator=True,
        )

    def chunk_documents(self, documents: List[Document]) -> tuple[List[Document], List[Document]]:
        """
        Create parent-child chunk hierarchy.
        
        Returns:
            Tuple of (parent_chunks, child_chunks)
            - Parents: Large (2000 chars) for full LLM context
            - Children: Small (500 chars) for precise retrieval
            - Children reference parents via parent_id in metadata
        """
        all_parents = []
        all_children = []
        
        for doc in documents:
            content = self._preprocess_content(doc.page_content)
            
            temp_doc = Document(
                page_content=content,
                metadata=doc.metadata.copy(),
            )
            
            parents = self.parent_splitter.split_documents([temp_doc])
            
            for parent_idx, parent in enumerate(parents):
                parent_id = f"{doc.metadata.get('source', 'unknown')}_parent_{parent_idx}"
                parent.metadata["parent_id"] = parent_id
                parent.metadata["chunk_type"] = "parent"
                parent.metadata["parent_index"] = parent_idx
                parent.metadata["total_parents"] = len(parents)
                
                section = self._extract_section_header(parent.page_content)
                if section:
                    parent.metadata["section"] = section
                
                all_parents.append(parent)
                
                children = self.child_splitter.split_documents([parent])
                
                for child_idx, child in enumerate(children):
                    child.metadata["parent_id"] = parent_id
                    child.metadata["chunk_type"] = "child"
                    child.metadata["child_index"] = child_idx
                    child.metadata["total_children"] = len(children)
                    child.metadata["chunk_size"] = len(child.page_content)
                    
                    child_section = self._extract_section_header(child.page_content)
                    if child_section:
                        child.metadata["section"] = child_section
                    
                    all_children.append(child)
            
            logger.info(
                f"[CHUNKER] {doc.metadata.get('source', 'unknown')}: "
                f"{len(doc.page_content):,} chars -> {len(parents)} parents, {len(all_children)} children"
            )
        
        logger.info(f"[CHUNKER] Total: {len(all_parents)} parents, {len(all_children)} children")
        return all_parents, all_children

    def _preprocess_content(self, content: str) -> str:
        """
        Clean and preprocess content before chunking.

        Args:
            content: Raw document content

        Returns:
            Cleaned content
        """
        content = re.sub(r'\n{4,}', '\n\n\n', content)

        content = re.sub(r'[ \t]+', ' ', content)

        lines = [line.rstrip() for line in content.split('\n')]
        content = '\n'.join(lines)

        return content.strip()

    def _extract_section_header(self, content: str) -> str:
        """
        Extract the first section header from chunk content.

        Args:
            content: Chunk content

        Returns:
            Section header or empty string
        """
        match = re.search(r'^#+\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        lines = content.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line and len(first_line) < 100:
                if first_line.isupper() or first_line.istitle():
                    return first_line

        return ""
