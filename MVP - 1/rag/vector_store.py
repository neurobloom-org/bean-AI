"""
Vector Store for RAG system
Uses ChromaDB for storing and retrieving therapeutic knowledge
"""

import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from config.settings import settings


class VectorStore:
    """
    Vector database for storing therapeutic knowledge
    Uses ChromaDB with sentence transformers for embeddings
    """
    
    def __init__(self):
        """
        Initialize vector store with ChromaDB
        """
        # Create directory if it doesn't exist
        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        print(f"[VectorStore] Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Get or create collection
        self.collection_name = "therapeutic_knowledge"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"[VectorStore] Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Therapeutic knowledge base"}
            )
            print(f"[VectorStore] Created new collection: {self.collection_name}")
    
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Add documents to the vector store
        
        Args:
            documents: List of text documents
            metadatas: Optional list of metadata dicts for each document
        """
        if not documents:
            print("[VectorStore] No documents to add")
            return
        
        print(f"[VectorStore] Adding {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Generate IDs
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{"source": "manual"} for _ in documents]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[VectorStore] Added {len(documents)} documents successfully")
    
    
    def add_documents_from_file(self, file_path: str):
        """
        Load and add documents from a text file
        Splits by paragraphs
        
        Args:
            file_path: Path to text file
        """
        try:
            print(f"[VectorStore] Loading documents from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks
            chunks = self._split_into_chunks(content)
            
            # Create metadata
            metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(chunks))]
            
            # Add to store
            self.add_documents(chunks, metadatas)
            
        except Exception as e:
            print(f"[VectorStore] Error loading file: {e}")
    
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - chunk_overlap
        
        return chunks
    
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        print(f"[VectorStore] Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        print(f"[VectorStore] Found {len(formatted_results)} relevant documents")
        return formatted_results
    
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'document_count': count,
            'embedding_model': settings.EMBEDDING_MODEL
        }
    
    
    def clear_collection(self):
        """
        Clear all documents from the collection
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Therapeutic knowledge base"}
        )
        print("[VectorStore] Collection cleared")


# Test function
if __name__ == "__main__":
    print("Testing VectorStore...")
    store = VectorStore()
    
    # Sample therapeutic knowledge
    sample_docs = [
        "Anxiety is a normal response to stress. Deep breathing exercises can help reduce anxiety symptoms.",
        "Depression is characterized by persistent sadness and loss of interest. Professional help is important.",
        "Cognitive Behavioral Therapy (CBT) is effective for treating anxiety and depression.",
        "Mindfulness meditation can reduce stress and improve emotional regulation.",
        "Exercise releases endorphins which can improve mood and reduce anxiety."
    ]
    
    print("\nAdding sample documents...")
    store.add_documents(sample_docs)
    
    print("\nSearching for 'anxiety'...")
    results = store.search("How to deal with anxiety?", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text']}")
    
    print("\n" + "="*60)
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")