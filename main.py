"""
RAG Demo: Homeowner Compliance Q&A
===================================
A minimal RAG pipeline for answering compliance questions using
city ordinances and HOA documents.

Run: python main.py
Requires: OPENAI_API_KEY environment variable
"""

import os
import re
from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI
import pdfplumber

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

CITY_ORDINANCE_PATH = "data/city_ordinance.txt"
HOA_PDF_PATH = "data/hoa_ccrs.pdf"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5  # Number of chunks to retrieve
MIN_SOURCES = 2  # Minimum sources required for a valid answer

# Initialize OpenAI client (uses OPENAI_API_KEY env var)
client = OpenAI()


# -----------------------------------------------------------------------------
# STEP 1: DOCUMENT LOADING
# -----------------------------------------------------------------------------

def load_city_ordinance(filepath: str) -> str:
    """Load city ordinance from a plain text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_hoa_pdf(filepath: str) -> str:
    """Extract text from HOA PDF using pdfplumber."""
    text_content = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)


# -----------------------------------------------------------------------------
# STEP 2: CHUNKING WITH METADATA
# -----------------------------------------------------------------------------

def chunk_city_ordinance(text: str) -> List[Dict]:
    """
    Split city ordinance into section-level chunks.
    Each 'Section X.XX.XXX - Title' starts a new chunk.
    """
    chunks = []
    
    # Pattern to match section headers like "Section 12.04.030 - Height Restrictions"
    section_pattern = r'(Section \d+\.\d+\.\d+ - [^\n]+)'
    
    # Split on section headers, keeping the headers
    parts = re.split(section_pattern, text)
    
    # First part is preamble (title, chapter info)
    preamble = parts[0].strip()
    if preamble:
        chunks.append({
            "text": preamble,
            "metadata": {
                "authority_level": "city",
                "section_id": "Title/Chapter Info",
                "source_name": "Maplewood Municipal Code"
            }
        })
    
    # Pair headers with their content
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            header = parts[i].strip()
            content = parts[i + 1].strip()
            
            # Extract section ID from header
            section_match = re.search(r'Section (\d+\.\d+\.\d+)', header)
            section_id = section_match.group(1) if section_match else "Unknown"
            
            chunks.append({
                "text": f"{header}\n{content}",
                "metadata": {
                    "authority_level": "city",
                    "section_id": section_id,
                    "source_name": "Maplewood Municipal Code"
                }
            })
    
    return chunks


def chunk_hoa_document(text: str) -> List[Dict]:
    """
    Split HOA CC&Rs into section-level chunks.
    Each 'Section X.X - Title' starts a new chunk.
    """
    chunks = []
    
    # Pattern to match section headers like "Section 7.2 - Fence Requirements"
    section_pattern = r'(Section \d+\.\d+ - [^\n]+)'
    
    # Split on section headers
    parts = re.split(section_pattern, text)
    
    # First part is preamble (title info)
    preamble = parts[0].strip()
    if preamble:
        chunks.append({
            "text": preamble,
            "metadata": {
                "authority_level": "hoa",
                "section_id": "Article Header",
                "source_name": "Maplewood Estates HOA CC&Rs"
            }
        })
    
    # Pair headers with their content
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            header = parts[i].strip()
            content = parts[i + 1].strip()
            
            # Extract section ID from header
            section_match = re.search(r'Section (\d+\.\d+)', header)
            section_id = section_match.group(1) if section_match else "Unknown"
            
            chunks.append({
                "text": f"{header}\n{content}",
                "metadata": {
                    "authority_level": "hoa",
                    "section_id": section_id,
                    "source_name": "Maplewood Estates HOA CC&Rs"
                }
            })
    
    return chunks


# -----------------------------------------------------------------------------
# STEP 3: EMBEDDING GENERATION
# -----------------------------------------------------------------------------

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """
    Generate embeddings for each chunk using OpenAI's embedding model.
    Stores embedding directly in each chunk dict.
    """
    # Extract texts for batch embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Call OpenAI embeddings API
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    
    # Attach embeddings to chunks
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = response.data[i].embedding
    
    return chunks


# -----------------------------------------------------------------------------
# STEP 4: IN-MEMORY VECTOR STORE
# -----------------------------------------------------------------------------

class SimpleVectorStore:
    """
    Minimal in-memory vector store using cosine similarity.
    No external database required.
    """
    
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        # Pre-compute embedding matrix for efficient similarity search
        self.embeddings = np.array([c["embedding"] for c in chunks])
        # Normalize for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Find top-k most similar chunks to the query.
        Returns chunks sorted by similarity (highest first).
        """
        # Normalize query embedding
        query_vec = np.array(query_embedding)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_vec)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunks with their similarity scores
        results = []
        for idx in top_indices:
            chunk_copy = self.chunks[idx].copy()
            chunk_copy["similarity"] = float(similarities[idx])
            results.append(chunk_copy)
        
        return results


# -----------------------------------------------------------------------------
# STEP 5: RETRIEVAL WITH AUTHORITY SORTING
# -----------------------------------------------------------------------------

def retrieve_and_sort(
    query: str,
    vector_store: SimpleVectorStore,
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve relevant chunks and sort by authority level.
    City ordinances take precedence over HOA rules.
    """
    # Generate embedding for the query
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    query_embedding = response.data[0].embedding
    
    # Retrieve top-k chunks
    results = vector_store.search(query_embedding, top_k)
    
    # Sort by authority: city first, then hoa
    authority_order = {"city": 0, "hoa": 1}
    results.sort(key=lambda x: authority_order.get(
        x["metadata"]["authority_level"], 2
    ))
    
    return results


# -----------------------------------------------------------------------------
# STEP 6: ANSWER GENERATION WITH GUARDRAILS
# -----------------------------------------------------------------------------

def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Generate an answer using the LLM based only on retrieved chunks.
    Includes guardrails for insufficient information.
    """
    # Guardrail: Check minimum sources
    if len(retrieved_chunks) < MIN_SOURCES:
        return "Insufficient information to answer this question reliably."
    
    # Build context from retrieved chunks
    context_parts = []
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        context_parts.append(
            f"[{meta['authority_level'].upper()}] {meta['source_name']} - "
            f"Section {meta['section_id']}:\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    # Build the prompt
    system_prompt = """You are a helpful assistant that answers homeowner compliance questions.

RULES:
1. ONLY use information from the provided sources
2. NEVER make up rules or requirements not in the sources
3. City ordinances take legal precedence over HOA rules
4. Always cite section IDs when stating requirements
5. If the sources don't contain enough information, say so

OUTPUT FORMAT:
Short Answer: [Direct yes/no or brief answer]

Explanation: [Detailed explanation with specific requirements]

Sources:
- [authority level] Section X.X - Brief description of what was cited"""

    user_prompt = f"""Question: {question}

Retrieved Sources:
{context}

Based ONLY on the sources above, answer the homeowner's question."""

    # Call the LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1  # Low temperature for factual responses
    )
    
    return response.choices[0].message.content


# -----------------------------------------------------------------------------
# MAIN: END-TO-END PIPELINE
# -----------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("HOA Compliance RAG Demo")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\n[1] Loading documents...")
    city_text = load_city_ordinance(CITY_ORDINANCE_PATH)
    hoa_text = load_hoa_pdf(HOA_PDF_PATH)
    print(f"    - City ordinance: {len(city_text)} characters")
    print(f"    - HOA CC&Rs: {len(hoa_text)} characters")
    
    # Step 2: Chunk documents
    print("\n[2] Chunking documents...")
    city_chunks = chunk_city_ordinance(city_text)
    hoa_chunks = chunk_hoa_document(hoa_text)
    all_chunks = city_chunks + hoa_chunks
    print(f"    - City chunks: {len(city_chunks)}")
    print(f"    - HOA chunks: {len(hoa_chunks)}")
    print(f"    - Total chunks: {len(all_chunks)}")
    
    # Step 3: Generate embeddings
    print("\n[3] Generating embeddings...")
    all_chunks = generate_embeddings(all_chunks)
    print(f"    - Embedded {len(all_chunks)} chunks")
    
    # Step 4: Create vector store
    print("\n[4] Creating in-memory vector store...")
    vector_store = SimpleVectorStore(all_chunks)
    print("    - Vector store ready")
    
    # Step 5: Example question
    print("\n[5] Processing question...")
    question = "Can I build a 6-foot fence in my backyard?"
    print(f"    Question: {question}")
    
    # Retrieve relevant chunks
    print("\n[6] Retrieving relevant chunks...")
    retrieved = retrieve_and_sort(question, vector_store, TOP_K)
    print(f"    - Retrieved {len(retrieved)} chunks:")
    for chunk in retrieved:
        meta = chunk["metadata"]
        print(f"      • [{meta['authority_level']}] Section {meta['section_id']} "
              f"(similarity: {chunk['similarity']:.3f})")
    
    # Generate answer
    print("\n[7] Generating answer...")
    print("-" * 60)
    answer = generate_answer(question, retrieved)
    print(answer)
    print("-" * 60)
    
    print("\n✓ RAG pipeline complete!")


if __name__ == "__main__":
    main()
