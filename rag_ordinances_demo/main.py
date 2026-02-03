#!/usr/bin/env python3
"""
RAG Ordinances Demo - Multi-Jurisdiction Compliance Q&A
=======================================================

A citation-first RAG system for answering homeowner compliance questions
using city ordinances and HOA documents with strict jurisdiction filtering.

SETUP
-----
1. Install dependencies:
   pip install -r requirements.txt

2. Generate sample HOA PDFs:
   python create_sample_pdfs.py

3. Set your OpenAI API key:
   export OPENAI_API_KEY="your-key-here"

USAGE
-----
# Ask a question for a specific city and HOA:
python main.py --city "san_diego" --hoa "demo_hoa_1" --question "Can I build a 6-foot fence?"

# Run the evaluation suite:
python eval.py

# List available cities and HOAs:
python main.py --list

AVAILABLE DATA
--------------
Cities: san_diego, los_angeles
HOAs: demo_hoa_1, demo_hoa_2

ARCHITECTURE
------------
- ingest.py   : Document loading and section-level chunking
- retrieve.py : Embeddings, in-memory index, jurisdiction filtering
- answer.py   : LLM answer generation with guardrails
- eval.py     : Evaluation harness with test cases

KEY FEATURES
------------
- Hard jurisdiction filtering (never retrieves from wrong city/HOA)
- Authority-sorted results (state > city > hoa)
- Explicit section ID citations
- Guardrails: minimum sources, no cross-jurisdiction leakage
- Deterministic section parsing (no "smart" LLM parsing)
"""

import argparse
import sys
from typing import Optional

from ingest import ingest_documents, list_available_cities, list_available_hoas
from retrieve import build_index, retrieve, validate_retrieval_jurisdiction
from answer import generate_answer


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

DEFAULT_TOP_K = 5


# -----------------------------------------------------------------------------
# CLI FUNCTIONS
# -----------------------------------------------------------------------------

def list_data():
    """List all available cities and HOAs."""
    cities = list_available_cities()
    hoas = list_available_hoas()
    
    print("\nAvailable Cities:")
    if cities:
        for city in cities:
            print(f"  - {city}")
    else:
        print("  (none found - run create_sample_pdfs.py first)")
    
    print("\nAvailable HOAs:")
    if hoas:
        for hoa in hoas:
            print(f"  - {hoa}")
    else:
        print("  (none found - run create_sample_pdfs.py first)")


def run_query(city: str, hoa: str, question: str, top_k: int = DEFAULT_TOP_K, verbose: bool = False):
    """
    Run a single query through the RAG pipeline.
    
    Args:
        city: City identifier (e.g., "san_diego")
        hoa: HOA identifier (e.g., "demo_hoa_1")
        question: User's question
        top_k: Number of chunks to retrieve
        verbose: Whether to show detailed retrieval info
    """
    print("="*60)
    print("RAG COMPLIANCE Q&A")
    print("="*60)
    print(f"City:     {city}")
    print(f"HOA:      {hoa}")
    print(f"Question: {question}")
    print("="*60)
    
    # Step 1: Ingest documents
    print("\n[1/4] Loading and chunking documents...")
    try:
        chunks = ingest_documents(city, hoa)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run 'python create_sample_pdfs.py' to generate sample data.")
        print("Use 'python main.py --list' to see available cities and HOAs.")
        sys.exit(1)
    
    print(f"      Total chunks: {len(chunks)}")
    
    # Step 2: Build index
    print("\n[2/4] Building vector index...")
    index = build_index(chunks)
    stats = index.get_stats()
    print(f"      Index stats: {stats['city_chunks']} city, {stats['hoa_chunks']} HOA chunks")
    
    # Step 3: Retrieve
    print(f"\n[3/4] Retrieving top-{top_k} relevant chunks...")
    retrieved = retrieve(question, index, city, hoa, top_k)
    print(f"      Retrieved {len(retrieved)} chunks")
    
    if verbose:
        print("\n      Retrieved chunks:")
        for i, chunk in enumerate(retrieved, 1):
            meta = chunk["metadata"]
            print(f"      {i}. [{meta['authority_level']}] {meta['section_id']} "
                  f"(sim: {chunk['similarity']:.3f})")
    
    # Validate jurisdiction
    is_valid, error = validate_retrieval_jurisdiction(retrieved, city, hoa)
    if not is_valid:
        print(f"\n      WARNING: {error}")
    
    # Step 4: Generate answer
    print("\n[4/4] Generating answer...")
    answer = generate_answer(question, retrieved, city, hoa)
    
    # Print answer
    print("\n" + "-"*60)
    print("ANSWER")
    print("-"*60)
    print(answer)
    print("-"*60)
    
    return answer


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG Compliance Q&A - Multi-jurisdiction homeowner compliance assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --city san_diego --hoa demo_hoa_1 --question "Can I build a 6-foot fence?"
  python main.py --city los_angeles --hoa demo_hoa_2 -q "What materials are prohibited?"
  python main.py --list
  python eval.py
        """
    )
    
    parser.add_argument(
        "--city", "-c",
        type=str,
        help="City identifier (e.g., 'san_diego', 'los_angeles')"
    )
    
    parser.add_argument(
        "--hoa", "-o",
        type=str,
        help="HOA identifier (e.g., 'demo_hoa_1', 'demo_hoa_2')"
    )
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Question to ask about compliance"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed retrieval information"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available cities and HOAs"
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_data()
        return
    
    # Validate required arguments for query
    if not args.city or not args.hoa or not args.question:
        parser.print_help()
        print("\nERROR: --city, --hoa, and --question are required for queries.")
        print("Use --list to see available options.")
        sys.exit(1)
    
    # Run the query
    run_query(
        city=args.city,
        hoa=args.hoa,
        question=args.question,
        top_k=args.top_k,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
