"""
Generate sample HOA CC&Rs PDFs for testing the multi-jurisdiction RAG demo.
Run: python create_sample_pdfs.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import os


def create_hoa_pdf(output_path: str, hoa_name: str, content_sections: list):
    """Generate a PDF with the given HOA content."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=14, spaceAfter=12)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'], fontSize=11, spaceAfter=6, spaceBefore=12)
    body_style = styles['Normal']
    
    content = []
    content.append(Paragraph(hoa_name, title_style))
    content.append(Paragraph("Declaration of Covenants, Conditions, and Restrictions", styles['Heading3']))
    content.append(Spacer(1, 0.25 * inch))
    
    for title, text in content_sections:
        content.append(Paragraph(title, section_style))
        content.append(Paragraph(text, body_style))
        content.append(Spacer(1, 0.1 * inch))
    
    doc.build(content)
    print(f"Created {output_path}")


def main():
    # Demo HOA 1 - Coastal Community (San Diego area)
    demo_hoa_1_sections = [
        ("ARTICLE 7: ARCHITECTURAL CONTROL",
         "All exterior modifications require prior written approval from the Architectural Review Committee (ARC). Applications must be submitted at least 30 days before planned construction."),
        
        ("Section 7.1 - Fence Height Requirements",
         "Maximum fence heights in Coastal Villas: (a) Front yard: 4 feet maximum, decorative styles only. (b) Side yard: 6 feet maximum. (c) Rear yard: 6 feet maximum. (d) Pool enclosures must be at least 5 feet high per California law."),
        
        ("Section 7.2 - Approved Fence Materials",
         "Approved materials: White vinyl, natural cedar, painted wood in approved colors, tubular steel in black or bronze. Prohibited: Chain link (all areas), barbed wire, corrugated metal, bamboo, and unpainted wood."),
        
        ("Section 7.3 - Fence Colors",
         "Pre-approved colors: White, Desert Sand, Coastal Gray, Natural Cedar Stain, and Black (iron/steel only). Other colors require ARC approval with a minimum 4-week review period."),
        
        ("Section 7.4 - ARC Application Process",
         "To install or modify a fence: (1) Submit ARC Form F-1 with site plan. (2) Include material samples and color chips. (3) Pay $50 application fee. (4) Allow 30 days for review. (5) Approval valid for 120 days."),
        
        ("Section 7.5 - Shared Fence Responsibility",
         "For fences on property lines: Both owners share maintenance costs equally. Disputes shall be submitted to the HOA Board for binding mediation. No owner may unilaterally remove or modify a shared fence."),
        
        ("Section 7.6 - Maintenance Standards",
         "All fences must be maintained in good repair. Owners must: (a) Repair damage within 21 days of notice. (b) Repaint/restain every 5 years minimum. (c) Keep fences free of vines unless pre-approved."),
        
        ("Section 7.7 - Violations and Fines",
         "Fence violations: First offense - Written warning, 30 days to cure. Second offense - $75 fine. Third offense - $150 fine plus mandatory removal. Continued non-compliance may result in liens."),
        
        ("Section 7.8 - Variances",
         "Homeowners may request height or material variances for documented hardship. Variance requests require: Written explanation, neighbor signatures within 50 feet, and ARC hearing attendance.")
    ]
    
    create_hoa_pdf(
        "data/hoas/demo_hoa_1/ccrs.pdf",
        "COASTAL VILLAS HOMEOWNERS ASSOCIATION",
        demo_hoa_1_sections
    )
    
    # Demo HOA 2 - Mountain Community (broader LA area)
    demo_hoa_2_sections = [
        ("ARTICLE 5: PROPERTY STANDARDS",
         "This Article governs exterior modifications, landscaping, and structural additions. All changes visible from common areas or neighboring properties require Design Review Board (DRB) approval."),
        
        ("Section 5.1 - General Fence Requirements",
         "Fences serve privacy and security purposes but must maintain community aesthetics. All fences require DRB approval before installation. Emergency repairs may proceed but must be reported within 72 hours."),
        
        ("Section 5.2 - Fence Height Limits",
         "Height maximums at Mountain Ridge Estates: (a) Front setback area: 3 feet (decorative only). (b) Side yards: 6 feet solid, up to 8 feet with lattice top. (c) Rear yards: 6 feet standard, 8 feet if property borders open space."),
        
        ("Section 5.3 - Material Requirements",
         "Required materials by location: Front yards - wrought iron or white picket only. Side/rear yards - wood, vinyl, or iron permitted. Prohibited everywhere: chain link, wire mesh, plastic sheeting, and cinder block (unless stuccoed)."),
        
        ("Section 5.4 - Color Palette",
         "Approved fence colors: Earth tones (Tan, Brown, Terracotta), Forest Green, Black, and White. Wood must be stained or painted within 90 days of installation. Raw wood is not permitted."),
        
        ("Section 5.5 - Wildlife Considerations",
         "Properties bordering natural areas: Bottom of fence must allow 6-inch gap for wildlife passage OR include wildlife gates every 50 feet. Solid barriers to natural areas require Environmental Review."),
        
        ("Section 5.6 - DRB Approval Process",
         "Submit DRB Form DR-3 with: Scaled drawings, material specifications, photos of property, and $35 review fee. Standard review: 21 days. Expedited review (additional $50): 10 business days."),
        
        ("Section 5.7 - Good Neighbor Provisions",
         "Finished side of fence must face outward. Shared fences require written agreement between neighbors before DRB submission. HOA mediates disputes at no charge for first mediation."),
        
        ("Section 5.8 - Enforcement",
         "Violations subject to: Warning (14 days to respond), $100 fine, $200 fine, mandatory hearing. Board may authorize removal at owner's expense after third violation.")
    ]
    
    create_hoa_pdf(
        "data/hoas/demo_hoa_2/ccrs.pdf",
        "MOUNTAIN RIDGE ESTATES HOMEOWNERS ASSOCIATION",
        demo_hoa_2_sections
    )
    
    print("\nSample PDFs created successfully!")


if __name__ == "__main__":
    main()
