"""
RAG Retriever
Retrieves relevant context from vector store for therapeutic responses
"""

from typing import List, Dict
from .vector_store import VectorStore


class RAGRetriever:
    """
    Retrieves relevant therapeutic knowledge from vector store
    Formats context for LLM consumption
    """
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: VectorStore instance (creates new if None)
        """
        self.vector_store = vector_store or VectorStore()
        print("[RAG Retriever] Initialized")
    
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User's query
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        print(f"[RAG Retriever] Retrieving context for: '{query}'")
        
        # Search vector store
        results = self.vector_store.search(query, top_k=top_k)
        
        if not results:
            print("[RAG Retriever] No relevant context found")
            return ""
        
        # Format context
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Context {i}]")
            context_parts.append(result['text'])
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        print(f"[RAG Retriever] Retrieved {len(results)} context chunks")
        return context
    
    
    def retrieve_with_metadata(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve context with full metadata
        
        Args:
            query: User's query
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents with metadata
        """
        results = self.vector_store.search(query, top_k=top_k)
        return results
    
    
    def format_context_for_llm(self, query: str, top_k: int = 3) -> str:
        """
        Format retrieved context specifically for LLM prompt
        
        Args:
            query: User's query
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string for LLM
        """
        context = self.retrieve_context(query, top_k)
        
        if not context:
            return "No specific therapeutic context available. Use general supportive guidance."
        
        formatted = f"""Based on the following therapeutic knowledge:

{context}

Please provide an empathetic, evidence-based response to the user."""
        
        return formatted


# Test function
if __name__ == "__main__":
    print("Testing RAG Retriever...")
    
    # Create vector store and add sample data
    from rag.vector_store import VectorStore
    
    store = VectorStore()
    
    sample_docs = [
        "Anxiety disorders are among the most common mental health conditions. Symptoms include excessive worry, restlessness, and physical tension. Treatment options include therapy, medication, and lifestyle changes.",
        "Cognitive Behavioral Therapy (CBT) helps identify and change negative thought patterns. It's highly effective for anxiety and depression. CBT teaches practical coping skills.",
        "Mindfulness meditation involves focusing on the present moment without judgment. Research shows it reduces stress, anxiety, and improves emotional regulation.",
        "Deep breathing exercises activate the parasympathetic nervous system, which calms the body. Try the 4-7-8 technique: breathe in for 4, hold for 7, exhale for 8.",
        "Depression symptoms include persistent sadness, loss of interest, fatigue, and changes in sleep or appetite. Professional help is important. Treatment is effective."
    ]
    
    print("\nAdding sample therapeutic knowledge...")
    store.add_documents(sample_docs)
    
    # Test retrieval
    retriever = RAGRetriever(vector_store=store)
    
    test_queries = [
        "I'm feeling anxious all the time",
        "How can I calm down when stressed?",
        "I think I might be depressed"
    ]
    
    print("\n" + "="*60)
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-"*60)
        context = retriever.retrieve_context(query, top_k=2)
        print(context)
        print("="*60)