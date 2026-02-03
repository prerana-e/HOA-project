"""
ingest.py - Document Loading and Chunking
=========================================
Handles loading city ordinances (text) and HOA CC&Rs (PDF),
splitting them into section-level chunks with rich metadata.
"""

import os
import re
from typing import List, Dict, Optional
from pathlib import Path
import pdfplumber


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
CITIES_DIR = DATA_DIR / "cities"
HOAS_DIR = DATA_DIR / "hoas"


# -----------------------------------------------------------------------------
# DOCUMENT LOADING
# -----------------------------------------------------------------------------

def load_city_ordinance(city: str) -> tuple[str, str]:
    """
    Load city ordinance text file.
    
    Args:
        city: City identifier (e.g., "san_diego")
    
    Returns:
        Tuple of (text_content, file_path)
    
    Raises:
        FileNotFoundError if city ordinance doesn't exist
    """
    ordinance_path = CITIES_DIR / city / "ordinance.txt"
    if not ordinance_path.exists():
        raise FileNotFoundError(f"City ordinance not found: {ordinance_path}")
    
    with open(ordinance_path, 'r', encoding='utf-8') as f:
        return f.read(), str(ordinance_path)


def load_hoa_pdf(hoa: str) -> tuple[str, str]:
    """
    Extract text from HOA CC&Rs PDF using pdfplumber.
    
    Args:
        hoa: HOA identifier (e.g., "demo_hoa_1")
    
    Returns:
        Tuple of (text_content, file_path)
    
    Raises:
        FileNotFoundError if HOA PDF doesn't exist
    """
    pdf_path = HOAS_DIR / hoa / "ccrs.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"HOA CC&Rs not found: {pdf_path}")
    
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    
    return "\n".join(text_parts), str(pdf_path)


# -----------------------------------------------------------------------------
# SECTION SPLITTING - CITY ORDINANCES
# -----------------------------------------------------------------------------

def split_city_ordinance(text: str, city: str, source_path: str) -> List[Dict]:
    """
    Split city ordinance into section-level chunks.
    
    Splitting strategy (deterministic, pattern-based):
    - Primary: § followed by numbers (e.g., §142.0520)
    - Secondary: SEC. or Section followed by numbers
    - Tertiary: ALL CAPS headings on their own line
    
    Args:
        text: Full ordinance text
        city: City identifier for metadata
        source_path: Path to source file
    
    Returns:
        List of chunk dicts with text and metadata
    """
    chunks = []
    
    # Pattern matches § sections (e.g., §142.0520) or SEC./Section patterns
    # This regex captures the section header and allows us to split on it
    section_pattern = r'(§[\d.]+\s+[A-Z][A-Z\s\-]+|SEC\.\s*[\d.A-Z]+\s+[A-Z][A-Z\s\-]+|Section\s+[\d.]+\s*[-–]\s*[^\n]+)'
    
    parts = re.split(section_pattern, text, flags=re.IGNORECASE)
    
    # First part is preamble (title, chapter info)
    preamble = parts[0].strip()
    if preamble and len(preamble) > 50:  # Only include substantial preambles
        chunks.append(_create_chunk(
            text=preamble,
            jurisdiction_city=city,
            authority_level="city",
            doc_type="ordinance",
            section_id="PREAMBLE",
            source_path=source_path
        ))
    
    # Process section pairs (header + content)
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
            
        header = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        
        # Extract section ID from header
        section_id = _extract_section_id(header, pattern_type="city")
        
        full_text = f"{header}\n{content}" if content else header
        
        if len(full_text) > 20:  # Skip trivially small chunks
            chunks.append(_create_chunk(
                text=full_text,
                jurisdiction_city=city,
                authority_level="city",
                doc_type="ordinance",
                section_id=section_id,
                source_path=source_path
            ))
    
    return chunks


def _extract_section_id(header: str, pattern_type: str) -> str:
    """
    Extract section identifier from a header string.
    
    Args:
        header: Section header text
        pattern_type: "city" or "hoa" to use appropriate patterns
    
    Returns:
        Extracted section ID or "UNKNOWN"
    """
    if pattern_type == "city":
        # Match §142.0520 style
        match = re.search(r'§([\d.]+)', header)
        if match:
            return f"§{match.group(1)}"
        
        # Match SEC. 12.22.A.3 style
        match = re.search(r'SEC\.\s*([\d.A-Z]+)', header, re.IGNORECASE)
        if match:
            return f"SEC.{match.group(1)}"
        
        # Match Section X.X style
        match = re.search(r'Section\s*([\d.]+)', header, re.IGNORECASE)
        if match:
            return f"Section {match.group(1)}"
    
    elif pattern_type == "hoa":
        # Match ARTICLE X style
        match = re.search(r'ARTICLE\s*(\d+)', header, re.IGNORECASE)
        if match:
            return f"ARTICLE {match.group(1)}"
        
        # Match Section X.X style
        match = re.search(r'Section\s*([\d.]+)', header, re.IGNORECASE)
        if match:
            return f"Section {match.group(1)}"
    
    return "UNKNOWN"


# -----------------------------------------------------------------------------
# SECTION SPLITTING - HOA DOCUMENTS
# -----------------------------------------------------------------------------

def split_hoa_document(text: str, hoa: str, source_path: str, city: str) -> List[Dict]:
    """
    Split HOA CC&Rs into section-level chunks.
    
    Splitting strategy (deterministic, pattern-based):
    - Primary: ARTICLE followed by number
    - Secondary: Section followed by number (e.g., Section 7.1)
    - Tertiary: § symbol with numbers
    - Fallback: Numbered subsections like (a), (1), etc.
    
    Args:
        text: Full CC&Rs text
        hoa: HOA identifier for metadata
        source_path: Path to source file
        city: Associated city for jurisdiction filtering
    
    Returns:
        List of chunk dicts with text and metadata
    """
    chunks = []
    
    # Pattern for ARTICLE or Section headers
    section_pattern = r'(ARTICLE\s+\d+[:\s]+[^\n]+|Section\s+[\d.]+\s*[-–]\s*[^\n]+)'
    
    parts = re.split(section_pattern, text, flags=re.IGNORECASE)
    
    # First part is preamble (HOA name, declaration title)
    preamble = parts[0].strip()
    if preamble and len(preamble) > 30:
        chunks.append(_create_chunk(
            text=preamble,
            jurisdiction_city=city,
            authority_level="hoa",
            doc_type="ccrs",
            section_id="PREAMBLE",
            source_path=source_path,
            hoa_id=hoa
        ))
    
    # Process section pairs
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
            
        header = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        
        section_id = _extract_section_id(header, pattern_type="hoa")
        
        full_text = f"{header}\n{content}" if content else header
        
        if len(full_text) > 20:
            chunks.append(_create_chunk(
                text=full_text,
                jurisdiction_city=city,
                authority_level="hoa",
                doc_type="ccrs",
                section_id=section_id,
                source_path=source_path,
                hoa_id=hoa
            ))
    
    # If no sections were found, fall back to page-based chunking
    if len(chunks) <= 1:
        chunks = _fallback_chunking(text, city, hoa, source_path)
    
    return chunks


def _fallback_chunking(text: str, city: str, hoa: str, source_path: str) -> List[Dict]:
    """
    Fallback chunking when section patterns aren't found.
    Splits on double newlines and creates page_n_chunk_i identifiers.
    """
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) > 50:  # Skip very short paragraphs
            chunks.append(_create_chunk(
                text=para,
                jurisdiction_city=city,
                authority_level="hoa",
                doc_type="ccrs",
                section_id=f"page_1_chunk_{i+1}",
                source_path=source_path,
                hoa_id=hoa
            ))
    
    return chunks


# -----------------------------------------------------------------------------
# CHUNK CREATION HELPER
# -----------------------------------------------------------------------------

def _create_chunk(
    text: str,
    jurisdiction_city: str,
    authority_level: str,
    doc_type: str,
    section_id: str,
    source_path: str,
    hoa_id: Optional[str] = None
) -> Dict:
    """
    Create a standardized chunk dictionary with metadata.
    
    Metadata fields:
    - jurisdiction_city: City this applies to (for filtering)
    - authority_level: "state", "city", or "hoa"
    - doc_type: "ordinance" or "ccrs"
    - section_id: Best-effort section identifier
    - source_path: Path to original document
    - hoa_id: HOA identifier (only for HOA docs)
    """
    metadata = {
        "jurisdiction_city": jurisdiction_city,
        "authority_level": authority_level,
        "doc_type": doc_type,
        "section_id": section_id,
        "source_path": source_path,
    }
    
    if hoa_id:
        metadata["hoa_id"] = hoa_id
    
    return {
        "text": text,
        "metadata": metadata
    }


# -----------------------------------------------------------------------------
# MAIN INGESTION FUNCTION
# -----------------------------------------------------------------------------

def ingest_documents(city: str, hoa: str) -> List[Dict]:
    """
    Load and chunk all documents for a city/HOA combination.
    
    Args:
        city: City identifier (e.g., "san_diego")
        hoa: HOA identifier (e.g., "demo_hoa_1")
    
    Returns:
        List of all chunks with metadata, ready for embedding
    """
    all_chunks = []
    
    # Load and chunk city ordinance
    print(f"  Loading city ordinance: {city}")
    city_text, city_path = load_city_ordinance(city)
    city_chunks = split_city_ordinance(city_text, city, city_path)
    all_chunks.extend(city_chunks)
    print(f"    → {len(city_chunks)} chunks")
    
    # Load and chunk HOA CC&Rs
    print(f"  Loading HOA CC&Rs: {hoa}")
    hoa_text, hoa_path = load_hoa_pdf(hoa)
    hoa_chunks = split_hoa_document(hoa_text, hoa, hoa_path, city)
    all_chunks.extend(hoa_chunks)
    print(f"    → {len(hoa_chunks)} chunks")
    
    return all_chunks


# -----------------------------------------------------------------------------
# UTILITY: LIST AVAILABLE DATA
# -----------------------------------------------------------------------------

def list_available_cities() -> List[str]:
    """Return list of available city identifiers."""
    if not CITIES_DIR.exists():
        return []
    return [d.name for d in CITIES_DIR.iterdir() if d.is_dir()]


def list_available_hoas() -> List[str]:
    """Return list of available HOA identifiers."""
    if not HOAS_DIR.exists():
        return []
    return [d.name for d in HOAS_DIR.iterdir() if d.is_dir()]


# -----------------------------------------------------------------------------
# TEST / DEBUG
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test of ingestion
    print("Testing ingestion...")
    print(f"Available cities: {list_available_cities()}")
    print(f"Available HOAs: {list_available_hoas()}")
    
    if list_available_cities() and list_available_hoas():
        city = list_available_cities()[0]
        hoa = list_available_hoas()[0]
        print(f"\nIngesting {city} + {hoa}:")
        chunks = ingest_documents(city, hoa)
        print(f"\nTotal chunks: {len(chunks)}")
        for chunk in chunks[:3]:
            print(f"  [{chunk['metadata']['authority_level']}] "
                  f"{chunk['metadata']['section_id']}: "
                  f"{chunk['text'][:60]}...")
