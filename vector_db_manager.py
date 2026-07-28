import os
import ollama
import uuid
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from md_manager import MarkdownManager

class EmbeddingEngine:
    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.model_name = model_name
        print(f"Embedding engine linked to Ollama model: {self.model_name}")

    def get_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        processed_texts = []
        for text in texts:
            if is_query:
                processed_texts.append(f"Represent this sentence for searching relevant passages: {text}")
            else:
                processed_texts.append(text)

        try:
            response = ollama.embed(
                model=self.model_name,
                input=processed_texts
            )
            return response["embeddings"]
            
        except Exception as e:
            print(f"Error calling Ollama embeddings: {e}")
            raise e


class MarkdownSplitter:
    @staticmethod
    def split_by_headers(content: str) -> List[Dict[str, str]]:
        lines = content.split("\n")
        chunks = []
        current_header = "General"
        current_chunk_lines = []

        header_pattern = re.compile(r"^(#{1,4})\s+(.*)$")

        for line in lines:
            match = header_pattern.match(line)
            if match:
                if current_chunk_lines:
                    chunks.append({
                        "header": current_header,
                        "text": "\n".join(current_chunk_lines).strip()
                    })
                    current_chunk_lines = []
                current_header = match.group(2).strip()
            else:
                current_chunk_lines.append(line)

        if current_chunk_lines:
            chunks.append({
                "header": current_header,
                "text": "\n".join(current_chunk_lines).strip()
            })

        return [c for c in chunks if c["text"]]


class VectorDBManager:
    def __init__(self, db_path: str = "./data/vector_db", collection_name: str = "aether_kb"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder = EmbeddingEngine()
        self.md_manager = MarkdownManager()

    def upsert_markdown_file(self, category: str, filename: str) -> None:
        try:
            metadata, content = self.md_manager.read_file(category, filename)
        except Exception as e:
            print(f"Failed to read file for indexing: {e}")
            return

        chunks = MarkdownSplitter.split_by_headers(content)
        if not chunks:
            return

        documents = []
        ids = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            doc_text = f"Section: {chunk['header']}\n\n{chunk['text']}"
            documents.append(doc_text)
            
            chunk_id = f"{category}_{filename}_{idx}"
            ids.append(chunk_id)

            chunk_metadata = {
                "category": category,
                "source_file": filename,
                "section_header": chunk["header"]
            }
            for key, val in metadata.items():
                if isinstance(val, (str, int, float, bool)):
                    chunk_metadata[f"meta_{key}"] = val
                elif isinstance(val, list):
                    chunk_metadata[f"meta_{key}"] = ", ".join(map(str, val))

            metadatas.append(chunk_metadata)

        embeddings = self.embedder.get_embeddings(documents)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        print(f"Indexed {len(documents)} chunks from {category}/{filename} into Vector DB.")

    def search(self, query: str, category_filter: Optional[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.embedder.get_embeddings([query])[0]
        
        where_filter = {}
        if category_filter:
            where_filter["category"] = category_filter

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where_filter if where_filter else None
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for i in range(len(docs)):
                formatted_results.append({
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": distances[i]
                })

        return formatted_results

    def index_directory(self, category: str) -> None:
        if category not in self.md_manager.directories:
            print(f"[Error] Unknown directory category: '{category}'")
            return
            
        dir_path = self.md_manager.directories[category]
        print(f"\n[Indexer] Recursively scanning directory: '{dir_path.resolve()}'")
        
        md_files = list(dir_path.rglob("*.md"))
        if not md_files:
            print(f"[Indexer] No Markdown (.md) files found under the '{category}' directory.")
            return
            
        print(f"[Indexer] Found {len(md_files)} files. Initiating bulk parsing...")
        
        for file_path in md_files:
            relative_name = file_path.relative_to(dir_path).as_posix().replace(".md", "")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                    
                metadata = {}
                content = raw_text.strip()
                if raw_text.startswith("---"):
                    parts = raw_text.split("---", 2)
                    if len(parts) >= 3:
                        import yaml
                        try:
                            metadata = yaml.safe_load(parts[1]) or {}
                            content = parts[2].strip()
                        except Exception:
                            pass
                            
                chunks = MarkdownSplitter.split_by_headers(content)
                if not chunks:
                    continue
                    
                documents = []
                ids = []
                metadatas = []
                
                for idx, chunk in enumerate(chunks):
                    doc_text = f"Section: {chunk['header']}\n\n{chunk['text']}"
                    documents.append(doc_text)
                    
                    chunk_id = f"{category}_{relative_name.replace('/', '_')}_{idx}"
                    ids.append(chunk_id)
                    
                    chunk_metadata = {
                        "category": category,
                        "source_file": relative_name,
                        "section_header": chunk["header"]
                    }
                    for key, val in metadata.items():
                        if isinstance(val, (str, int, float, bool)):
                            chunk_metadata[f"meta_{key}"] = val
                        elif isinstance(val, list):
                            chunk_metadata[f"meta_{key}"] = ", ".join(map(str, val))
                            
                    metadatas.append(chunk_metadata)
                    
                embeddings = self.embedder.get_embeddings(documents)
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                print(f"[Indexer] Successfully indexed: {relative_name} ({len(documents)} chunks)")
                
            except Exception as e:
                print(f"[Indexer Error] Failed to process file '{file_path}': {e}")
                
        print(f"[Indexer] Dynamic indexing complete for category: '{category}'.")

if __name__ == "__main__":
    db_mgr = VectorDBManager()
    db_mgr.index_directory("rules")
    db_mgr.index_directory("lore")

    # print("\n--- Phase 1: Indexing Files ---")
    # db_mgr.upsert_markdown_file("locations", "whispering_woods_entry")
    # db_mgr.upsert_markdown_file("actors", "barnaby_merchant")

    # print("\n--- Phase 2: Targeted Querying (Locations Only) ---")
    # loc_results = db_mgr.search(
    #     query="What does the forest smell like?", 
    #     category_filter="locations", 
    #     limit=2
    # )

    # for idx, r in enumerate(loc_results):
    #     print(f"\nResult #{idx+1} (Source: {r['metadata']['source_file']})")
    #     print(f"Match: {r['document'].strip()}")

    # print("\n--- Phase 3: Targeted Querying (Actors Only) ---")
    # actor_results = db_mgr.search(
    #     query="Tell me about the gnome's coat and inventory.", 
    #     category_filter="actors", 
    #     limit=2
    # )

    # for idx, r in enumerate(actor_results):
    #     print(f"\nResult #{idx+1} (Source: {r['metadata']['source_file']})")
    #     print(f"Match: {r['document'].strip()}")
