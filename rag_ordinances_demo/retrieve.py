"""
retrieve.py - Embeddings, Index, and Retrieval
===============================================
Handles embedding generation, in-memory vector storage,
jurisdiction filtering, and authority-sorted retrieval.
"""

import os
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 5

# Initialize OpenAI client (uses OPENAI_API_KEY env var)
client = OpenAI()


# -----------------------------------------------------------------------------
# EMBEDDING GENERATION
# -----------------------------------------------------------------------------

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """
    Generate embeddings for each chunk using OpenAI's embedding API.
    
    Modifies chunks in-place, adding 'embedding' field to each.
    
    Args:
        chunks: List of chunk dicts with 'text' field
    
    Returns:
        Same chunks with 'embedding' field added
    """
    if not chunks:
        return chunks
    
    # Extract texts for batch embedding (more efficient than one-by-one)
    texts = [chunk["text"] for chunk in chunks]
    
    # OpenAI allows up to 2048 inputs per request; batch if needed
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    # Attach embeddings to chunks
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding
    
    return chunks


# -----------------------------------------------------------------------------
# IN-MEMORY VECTOR INDEX
# -----------------------------------------------------------------------------

class VectorIndex:
    """
    In-memory vector index with jurisdiction filtering.
    
    Uses cosine similarity for retrieval and supports hard filtering
    by city and HOA to prevent cross-jurisdiction leakage.
    """
    
    def __init__(self, chunks: List[Dict]):
        """
        Initialize index with embedded chunks.
        
        Args:
            chunks: List of chunks, each must have 'embedding' and 'metadata'
        """
        self.chunks = chunks
        
        # Build embedding matrix for efficient similarity computation
        if chunks:
            self.embeddings = np.array([c["embedding"] for c in chunks])
            # Normalize vectors for cosine similarity (dot product of unit vectors)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            self.embeddings = self.embeddings / norms
        else:
            self.embeddings = np.array([])
    
    def search(
        self,
        query_embedding: List[float],
        city: str,
        hoa: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict]:
        """
        Search for relevant chunks with strict jurisdiction filtering.
        
        IMPORTANT: This implements hard filtering - we NEVER return chunks
        from other cities or HOAs, even if they're semantically similar.
        
        Args:
            query_embedding: Embedded query vector
            city: City to filter by (exact match required)
            hoa: HOA to filter by (exact match required)
            top_k: Maximum number of results to return
        
        Returns:
            List of matching chunks, sorted by similarity
        """
        if len(self.chunks) == 0:
            return []
        
        # Step 1: Hard filter by jurisdiction BEFORE similarity search
        # This ensures we NEVER retrieve from wrong city/HOA
        valid_indices = []
        for i, chunk in enumerate(self.chunks):
            meta = chunk["metadata"]
            
            # City documents: must match city
            if meta["authority_level"] == "city":
                if meta["jurisdiction_city"] == city:
                    valid_indices.append(i)
            
            # HOA documents: must match both city AND hoa
            elif meta["authority_level"] == "hoa":
                if (meta["jurisdiction_city"] == city and 
                    meta.get("hoa_id") == hoa):
                    valid_indices.append(i)
            
            # State documents: available to all (future use)
            elif meta["authority_level"] == "state":
                valid_indices.append(i)
        
        if not valid_indices:
            return []
        
        # Step 2: Compute similarities only for valid chunks
        query_vec = np.array(query_embedding)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Get embeddings for valid indices only
        valid_embeddings = self.embeddings[valid_indices]
        similarities = np.dot(valid_embeddings, query_vec)
        
        # Step 3: Get top-k from valid set
        top_k_local = min(top_k, len(valid_indices))
        local_top_indices = np.argsort(similarities)[::-1][:top_k_local]
        
        # Map back to original indices and build results
        results = []
        for local_idx in local_top_indices:
            original_idx = valid_indices[local_idx]
            chunk_copy = self.chunks[original_idx].copy()
            chunk_copy["similarity"] = float(similarities[local_idx])
            # Remove embedding from result (not needed downstream, saves memory)
            chunk_copy.pop("embedding", None)
            results.append(chunk_copy)
        
        return results
    
    def get_stats(self) -> Dict:
        """Return statistics about the index."""
        city_chunks = sum(1 for c in self.chunks 
                         if c["metadata"]["authority_level"] == "city")
        hoa_chunks = sum(1 for c in self.chunks 
                        if c["metadata"]["authority_level"] == "hoa")
        
        cities = set(c["metadata"]["jurisdiction_city"] for c in self.chunks)
        hoas = set(c["metadata"].get("hoa_id") for c in self.chunks 
                   if c["metadata"].get("hoa_id"))
        
        return {
            "total_chunks": len(self.chunks),
            "city_chunks": city_chunks,
            "hoa_chunks": hoa_chunks,
            "unique_cities": list(cities),
            "unique_hoas": list(hoas)
        }


# -----------------------------------------------------------------------------
# RETRIEVAL WITH AUTHORITY SORTING
# -----------------------------------------------------------------------------

def retrieve(
    question: str,
    index: VectorIndex,
    city: str,
    hoa: str,
    top_k: int = DEFAULT_TOP_K
) -> List[Dict]:
    """
    Retrieve relevant chunks for a question with authority sorting.
    
    Process:
    1. Generate embedding for question
    2. Search index with jurisdiction filtering
    3. Sort results by authority level (state > city > hoa)
    
    Args:
        question: User's question text
        index: VectorIndex to search
        city: City to filter by
        hoa: HOA to filter by
        top_k: Maximum chunks to retrieve
    
    Returns:
        List of relevant chunks, sorted by authority (state first, then city, then hoa)
    """
    # Generate embedding for the question
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question]
    )
    query_embedding = response.data[0].embedding
    
    # Search with jurisdiction filtering
    results = index.search(query_embedding, city, hoa, top_k)
    
    # Sort by authority level (state > city > hoa)
    # Lower number = higher authority
    authority_order = {
        "state": 0,
        "city": 1,
        "hoa": 2
    }
    
    results.sort(key=lambda x: authority_order.get(
        x["metadata"]["authority_level"], 99
    ))
    
    return results


# -----------------------------------------------------------------------------
# VALIDATION HELPERS
# -----------------------------------------------------------------------------

def validate_retrieval_jurisdiction(
    results: List[Dict],
    expected_city: str,
    expected_hoa: str
) -> tuple[bool, str]:
    """
    Validate that all retrieved chunks belong to expected jurisdiction.
    
    This is a safety check - if our filtering is correct, this should
    always pass. But we check anyway to catch bugs.
    
    Args:
        results: Retrieved chunks
        expected_city: City that should be in all results
        expected_hoa: HOA that should be in HOA results
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    for chunk in results:
        meta = chunk["metadata"]
        
        # Check city matches
        if meta["jurisdiction_city"] != expected_city:
            return (False, 
                    f"Cross-jurisdiction leakage detected: "
                    f"expected city '{expected_city}', "
                    f"got '{meta['jurisdiction_city']}'")
        
        # Check HOA matches (for HOA docs)
        if meta["authority_level"] == "hoa":
            if meta.get("hoa_id") != expected_hoa:
                return (False,
                        f"Cross-HOA leakage detected: "
                        f"expected HOA '{expected_hoa}', "
                        f"got '{meta.get('hoa_id')}'")
    
    return (True, "")


# -----------------------------------------------------------------------------
# BUILD INDEX FROM CHUNKS
# -----------------------------------------------------------------------------

def build_index(chunks: List[Dict]) -> VectorIndex:
    """
    Build a searchable index from chunks.
    
    Args:
        chunks: List of chunks from ingest.py
    
    Returns:
        VectorIndex ready for searching
    """
    print(f"  Generating embeddings for {len(chunks)} chunks...")
    embedded_chunks = generate_embeddings(chunks)
    print(f"  Building index...")
    index = VectorIndex(embedded_chunks)
    return index


# -----------------------------------------------------------------------------
# TEST / DEBUG
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test - requires ingest.py and data to be set up
    from ingest import ingest_documents, list_available_cities, list_available_hoas
    
    print("Testing retrieval...")
    
    cities = list_available_cities()
    hoas = list_available_hoas()
    
    if cities and hoas:
        city, hoa = cities[0], hoas[0]
        print(f"\nUsing: {city} + {hoa}")
        
        chunks = ingest_documents(city, hoa)
        index = build_index(chunks)
        
        print(f"\nIndex stats: {index.get_stats()}")
        
        question = "What is the maximum fence height in the backyard?"
        print(f"\nQuestion: {question}")
        
        results = retrieve(question, index, city, hoa, top_k=3)
        print(f"Retrieved {len(results)} chunks:")
        for r in results:
            meta = r["metadata"]
            print(f"  [{meta['authority_level']}] {meta['section_id']} "
                  f"(sim: {r['similarity']:.3f})")
        
        # Validate
        is_valid, error = validate_retrieval_jurisdiction(results, city, hoa)
        print(f"\nJurisdiction validation: {'PASS' if is_valid else 'FAIL'}")
        if not is_valid:
            print(f"  Error: {error}")
