"""
eval.py - Evaluation Harness for RAG Compliance Demo
====================================================
Runs test cases from tests/eval_cases.json and reports:
- Pass/fail per case
- Citation coverage score
- Cross-jurisdiction leakage detection

Usage: python eval.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from ingest import ingest_documents
from retrieve import build_index, retrieve, validate_retrieval_jurisdiction
from answer import generate_answer, extract_cited_section_ids


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

EVAL_CASES_PATH = Path(__file__).parent / "tests" / "eval_cases.json"
TOP_K = 5


# -----------------------------------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating a single test case."""
    case_id: str
    passed: bool
    must_cite_found: List[str]
    must_cite_missing: List[str]
    must_not_cite_found: List[str]  # Bad - these shouldn't be cited
    all_cited: List[str]
    jurisdiction_valid: bool
    jurisdiction_error: str
    answer_preview: str
    error: str = ""


# -----------------------------------------------------------------------------
# EVALUATION LOGIC
# -----------------------------------------------------------------------------

def load_eval_cases() -> List[Dict]:
    """Load test cases from JSON file."""
    with open(EVAL_CASES_PATH, 'r') as f:
        data = json.load(f)
    return data["cases"]


def normalize_section_id(section_id: str) -> str:
    """
    Normalize section ID for comparison.
    Handles variations like "§142.0520" vs "142.0520" vs "Section 142.0520"
    """
    # Remove common prefixes and normalize
    normalized = section_id.strip()
    normalized = normalized.replace("§", "").replace("SEC.", "").replace("Section", "")
    normalized = normalized.strip().upper()
    return normalized


def check_citation(cited_ids: List[str], expected_id: str) -> bool:
    """
    Check if expected_id appears in cited_ids (with normalization).
    """
    expected_norm = normalize_section_id(expected_id)
    
    for cited in cited_ids:
        cited_norm = normalize_section_id(cited)
        # Check if one contains the other (handles partial matches)
        if expected_norm in cited_norm or cited_norm in expected_norm:
            return True
    
    return False


def evaluate_case(
    case: Dict,
    indexes: Dict[Tuple[str, str], any]
) -> EvalResult:
    """
    Evaluate a single test case.
    
    Args:
        case: Test case dict from eval_cases.json
        indexes: Dict mapping (city, hoa) tuples to VectorIndex objects
    
    Returns:
        EvalResult with pass/fail and details
    """
    case_id = case["id"]
    city = case["city"]
    hoa = case["hoa"]
    question = case["question"]
    must_cite = case.get("must_cite", [])
    must_not_cite = case.get("must_not_cite", [])
    
    try:
        # Get or build index for this city/hoa combination
        key = (city, hoa)
        if key not in indexes:
            print(f"    Building index for {city}/{hoa}...")
            chunks = ingest_documents(city, hoa)
            indexes[key] = build_index(chunks)
        
        index = indexes[key]
        
        # Retrieve chunks
        retrieved = retrieve(question, index, city, hoa, TOP_K)
        
        # Validate jurisdiction
        jurisdiction_valid, jurisdiction_error = validate_retrieval_jurisdiction(
            retrieved, city, hoa
        )
        
        # Generate answer
        answer = generate_answer(question, retrieved, city, hoa)
        
        # Extract cited section IDs
        cited_ids = extract_cited_section_ids(answer)
        
        # Check must_cite
        must_cite_found = []
        must_cite_missing = []
        for expected in must_cite:
            if check_citation(cited_ids, expected):
                must_cite_found.append(expected)
            else:
                must_cite_missing.append(expected)
        
        # Check must_not_cite
        must_not_cite_found = []
        for forbidden in must_not_cite:
            if check_citation(cited_ids, forbidden):
                must_not_cite_found.append(forbidden)
        
        # Determine pass/fail
        # Pass if: all must_cite found, no must_not_cite found, jurisdiction valid
        passed = (
            len(must_cite_missing) == 0 and
            len(must_not_cite_found) == 0 and
            jurisdiction_valid
        )
        
        return EvalResult(
            case_id=case_id,
            passed=passed,
            must_cite_found=must_cite_found,
            must_cite_missing=must_cite_missing,
            must_not_cite_found=must_not_cite_found,
            all_cited=cited_ids,
            jurisdiction_valid=jurisdiction_valid,
            jurisdiction_error=jurisdiction_error,
            answer_preview=answer[:200] + "..." if len(answer) > 200 else answer
        )
        
    except Exception as e:
        return EvalResult(
            case_id=case_id,
            passed=False,
            must_cite_found=[],
            must_cite_missing=must_cite,
            must_not_cite_found=[],
            all_cited=[],
            jurisdiction_valid=False,
            jurisdiction_error="",
            answer_preview="",
            error=str(e)
        )


# -----------------------------------------------------------------------------
# REPORTING
# -----------------------------------------------------------------------------

def print_result(result: EvalResult, verbose: bool = False):
    """Print a single evaluation result."""
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"\n{status} | {result.case_id}")
    
    if result.error:
        print(f"  ERROR: {result.error}")
        return
    
    # Citation details
    if result.must_cite_found:
        print(f"  ✓ Found required: {result.must_cite_found}")
    if result.must_cite_missing:
        print(f"  ✗ Missing required: {result.must_cite_missing}")
    if result.must_not_cite_found:
        print(f"  ✗ Found forbidden: {result.must_not_cite_found}")
    
    # Jurisdiction
    if not result.jurisdiction_valid:
        print(f"  ✗ Jurisdiction error: {result.jurisdiction_error}")
    
    if verbose:
        print(f"  All citations: {result.all_cited}")
        print(f"  Answer preview: {result.answer_preview[:100]}...")


def print_summary(results: List[EvalResult]):
    """Print overall evaluation summary."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    # Citation coverage
    total_must_cite = sum(
        len(r.must_cite_found) + len(r.must_cite_missing) 
        for r in results
    )
    found_must_cite = sum(len(r.must_cite_found) for r in results)
    coverage = found_must_cite / total_must_cite if total_must_cite > 0 else 0
    
    # Cross-jurisdiction leakage
    leakage_cases = [r for r in results if not r.jurisdiction_valid]
    forbidden_cited = [r for r in results if r.must_not_cite_found]
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total cases:     {total}")
    print(f"Passed:          {passed} ({100*passed/total:.1f}%)")
    print(f"Failed:          {failed} ({100*failed/total:.1f}%)")
    print(f"\nCitation coverage: {found_must_cite}/{total_must_cite} ({100*coverage:.1f}%)")
    
    print(f"\nCross-jurisdiction leakage: {len(leakage_cases)} case(s)")
    if leakage_cases:
        for r in leakage_cases:
            print(f"  - {r.case_id}: {r.jurisdiction_error}")
    
    print(f"\nForbidden citations found: {len(forbidden_cited)} case(s)")
    if forbidden_cited:
        for r in forbidden_cited:
            print(f"  - {r.case_id}: cited {r.must_not_cite_found}")
    
    print("="*60)
    
    return passed == total


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def run_evaluation(verbose: bool = False) -> bool:
    """
    Run full evaluation suite.
    
    Returns:
        True if all tests passed, False otherwise
    """
    print("="*60)
    print("RAG COMPLIANCE DEMO - EVALUATION HARNESS")
    print("="*60)
    
    # Load test cases
    print("\nLoading test cases...")
    cases = load_eval_cases()
    print(f"Found {len(cases)} test cases")
    
    # Index cache to avoid rebuilding for same city/hoa
    indexes = {}
    
    # Run evaluations
    print("\nRunning evaluations...")
    results = []
    
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] Evaluating: {case['id']}")
        print(f"    City: {case['city']}, HOA: {case['hoa']}")
        print(f"    Question: {case['question'][:50]}...")
        
        result = evaluate_case(case, indexes)
        results.append(result)
        print_result(result, verbose=verbose)
    
    # Print summary
    all_passed = print_summary(results)
    
    return all_passed


if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    success = run_evaluation(verbose=verbose)
    sys.exit(0 if success else 1)
