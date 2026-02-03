"""
answer.py - LLM Answer Generation with Guardrails
=================================================
Generates answers strictly from retrieved context,
with proper citations and safety guardrails.
"""

from typing import List, Dict, Optional
from openai import OpenAI
from retrieve import validate_retrieval_jurisdiction

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

LLM_MODEL = "gpt-4o-mini"
MIN_SOURCES_REQUIRED = 2  # Minimum chunks needed for a valid answer
TEMPERATURE = 0.1  # Low temperature for factual, consistent responses

# Initialize OpenAI client
client = OpenAI()


# -----------------------------------------------------------------------------
# GUARDRAILS
# -----------------------------------------------------------------------------

class AnswerError(Exception):
    """Raised when answer generation should be refused."""
    pass


def check_guardrails(
    retrieved_chunks: List[Dict],
    city: str,
    hoa: str
) -> Optional[str]:
    """
    Check all guardrails before generating an answer.
    
    Guardrails:
    1. Minimum source count
    2. No cross-jurisdiction chunks
    
    Args:
        retrieved_chunks: Chunks from retrieval
        city: Expected city
        hoa: Expected HOA
    
    Returns:
        None if all checks pass, error message string if failed
    """
    # Guardrail 1: Minimum sources
    if len(retrieved_chunks) < MIN_SOURCES_REQUIRED:
        return (f"Insufficient information to answer this question reliably. "
                f"Only {len(retrieved_chunks)} source(s) found, "
                f"minimum {MIN_SOURCES_REQUIRED} required.")
    
    # Guardrail 2: Jurisdiction validation
    is_valid, error_msg = validate_retrieval_jurisdiction(
        retrieved_chunks, city, hoa
    )
    if not is_valid:
        return (f"Error: {error_msg}. "
                f"Refusing to answer due to potential cross-jurisdiction contamination.")
    
    return None  # All checks passed


# -----------------------------------------------------------------------------
# CONTEXT FORMATTING
# -----------------------------------------------------------------------------

def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a structured context string for the LLM.
    
    Format per chunk:
    [AUTHORITY_LEVEL] City: city_name | Section: section_id
    Source: source_path
    ---
    chunk text
    ===
    """
    parts = []
    
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        
        header = (
            f"[{meta['authority_level'].upper()}] "
            f"City: {meta['jurisdiction_city']} | "
            f"Section: {meta['section_id']}"
        )
        
        source_line = f"Source: {meta['source_path']}"
        
        parts.append(f"{header}\n{source_line}\n---\n{chunk['text']}")
    
    return "\n\n===\n\n".join(parts)


def format_sources(chunks: List[Dict]) -> str:
    """
    Format source citations for the answer output.
    
    Format: - [authority_level] [city] [section_id] (source_path)
    """
    lines = []
    seen = set()  # Deduplicate sources
    
    for chunk in chunks:
        meta = chunk["metadata"]
        
        # Create unique key for deduplication
        key = (meta["authority_level"], meta["jurisdiction_city"], 
               meta["section_id"], meta["source_path"])
        
        if key not in seen:
            seen.add(key)
            lines.append(
                f"- [{meta['authority_level']}] [{meta['jurisdiction_city']}] "
                f"{meta['section_id']} ({meta['source_path']})"
            )
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# ANSWER GENERATION
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a compliance assistant that helps homeowners understand regulations.

CRITICAL RULES:
1. ONLY use information explicitly stated in the provided sources
2. NEVER invent, assume, or extrapolate rules not in the sources
3. City ordinances take legal precedence over HOA rules when they conflict
4. Always cite specific section IDs when stating requirements
5. Use phrases like "based on the provided sources" - never give legal advice
6. If sources are unclear or incomplete, say so explicitly

AUTHORITY HIERARCHY (highest to lowest):
1. State law (if present)
2. City ordinances
3. HOA CC&Rs

OUTPUT FORMAT (use exactly this structure):
Short Answer:
[Direct yes/no or brief answer to the question]

Explanation:
[Detailed explanation citing specific sections. Mention both city and HOA requirements if relevant.]

Sources:
[Will be provided separately - do not generate this section]"""


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
    city: str,
    hoa: str
) -> str:
    """
    Generate an answer using LLM based strictly on retrieved context.
    
    Args:
        question: User's question
        retrieved_chunks: Chunks from retrieval (already filtered & sorted)
        city: City context
        hoa: HOA context
    
    Returns:
        Formatted answer string with Short Answer, Explanation, and Sources
    
    Raises:
        AnswerError: If guardrails fail
    """
    # Check guardrails first
    guardrail_error = check_guardrails(retrieved_chunks, city, hoa)
    if guardrail_error:
        return guardrail_error
    
    # Format context for LLM
    context = format_context(retrieved_chunks)
    
    # Build user prompt
    user_prompt = f"""Question from homeowner in {city} (HOA: {hoa}):
{question}

Retrieved Sources:
{context}

Based ONLY on the sources above, provide an answer following the required format.
Remember: City ordinances take precedence over HOA rules."""

    # Call LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=1000
    )
    
    llm_answer = response.choices[0].message.content
    
    # Append properly formatted sources
    sources = format_sources(retrieved_chunks)
    
    # Combine LLM answer with our formatted sources
    # (We format sources ourselves to ensure consistency)
    if "Sources:" in llm_answer:
        # Remove LLM's sources section if it generated one
        llm_answer = llm_answer.split("Sources:")[0].strip()
    
    final_answer = f"{llm_answer}\n\nSources:\n{sources}"
    
    return final_answer


# -----------------------------------------------------------------------------
# STRUCTURED ANSWER PARSING (for evaluation)
# -----------------------------------------------------------------------------

def parse_answer(answer_text: str) -> Dict:
    """
    Parse a formatted answer into structured components.
    
    Useful for evaluation to extract citations.
    
    Returns:
        Dict with keys: short_answer, explanation, sources (list)
    """
    result = {
        "short_answer": "",
        "explanation": "",
        "sources": [],
        "raw": answer_text
    }
    
    # Extract short answer
    if "Short Answer:" in answer_text:
        parts = answer_text.split("Short Answer:", 1)
        if len(parts) > 1:
            remainder = parts[1]
            if "Explanation:" in remainder:
                result["short_answer"] = remainder.split("Explanation:")[0].strip()
            else:
                result["short_answer"] = remainder.split("\n")[0].strip()
    
    # Extract explanation
    if "Explanation:" in answer_text:
        parts = answer_text.split("Explanation:", 1)
        if len(parts) > 1:
            remainder = parts[1]
            if "Sources:" in remainder:
                result["explanation"] = remainder.split("Sources:")[0].strip()
            else:
                result["explanation"] = remainder.strip()
    
    # Extract sources
    if "Sources:" in answer_text:
        sources_section = answer_text.split("Sources:", 1)[1].strip()
        for line in sources_section.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                result["sources"].append(line[1:].strip())
    
    return result


def extract_cited_section_ids(answer_text: str) -> List[str]:
    """
    Extract section IDs from the sources section of an answer.
    
    Returns list of section IDs like ["§142.0520", "Section 7.1"]
    """
    import re
    
    section_ids = []
    parsed = parse_answer(answer_text)
    
    for source_line in parsed["sources"]:
        # Pattern: [authority] [city] SECTION_ID (path)
        # Section ID is between the second ] and the (
        match = re.search(r'\]\s*\[.*?\]\s*(.+?)\s*\(', source_line)
        if match:
            section_id = match.group(1).strip()
            section_ids.append(section_id)
    
    return section_ids


# -----------------------------------------------------------------------------
# TEST / DEBUG
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test with mock chunks
    mock_chunks = [
        {
            "text": "§142.0520 HEIGHT LIMITS\n(c) REAR YARD: Fences may be up to six (6) feet in height in rear yards.",
            "metadata": {
                "jurisdiction_city": "san_diego",
                "authority_level": "city",
                "doc_type": "ordinance",
                "section_id": "§142.0520",
                "source_path": "data/cities/san_diego/ordinance.txt"
            },
            "similarity": 0.89
        },
        {
            "text": "Section 7.1 - Fence Height Requirements\nMaximum fence heights: (c) Rear yard: 6 feet maximum.",
            "metadata": {
                "jurisdiction_city": "san_diego",
                "authority_level": "hoa",
                "doc_type": "ccrs",
                "section_id": "Section 7.1",
                "source_path": "data/hoas/demo_hoa_1/ccrs.pdf",
                "hoa_id": "demo_hoa_1"
            },
            "similarity": 0.85
        }
    ]
    
    print("Testing answer generation...")
    
    question = "Can I build a 6-foot fence in my backyard?"
    answer = generate_answer(question, mock_chunks, "san_diego", "demo_hoa_1")
    
    print(f"\nQuestion: {question}")
    print("\n" + "="*60)
    print(answer)
    print("="*60)
    
    print("\n\nExtracted section IDs:", extract_cited_section_ids(answer))
