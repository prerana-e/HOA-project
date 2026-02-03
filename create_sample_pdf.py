"""
Generate a sample HOA CC&Rs PDF for testing the RAG demo.
Run this once to create data/hoa_ccrs.pdf
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def create_hoa_pdf():
    doc = SimpleDocTemplate(
        "data/hoa_ccrs.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12
    )
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=12
    )
    body_style = styles['Normal']
    
    content = []
    
    # Title
    content.append(Paragraph("MAPLEWOOD ESTATES HOMEOWNERS ASSOCIATION", title_style))
    content.append(Paragraph("Declaration of Covenants, Conditions, and Restrictions (CC&Rs)", styles['Heading2']))
    content.append(Spacer(1, 0.25 * inch))
    
    # Article 1
    content.append(Paragraph("ARTICLE 7: ARCHITECTURAL STANDARDS", section_style))
    content.append(Spacer(1, 0.1 * inch))
    
    sections = [
        ("Section 7.1 - Architectural Review Committee (ARC)",
         "All exterior modifications, including fences, must be submitted to the Architectural Review Committee for approval prior to construction. Submissions must include detailed plans, materials list, and color samples. The ARC will respond within 30 days of receiving a complete application."),
        
        ("Section 7.2 - Fence Requirements",
         "Fences must complement the aesthetic character of the community. All fences in Maplewood Estates must meet the following requirements: (a) Maximum height of six (6) feet for rear and side yards. (b) Maximum height of four (4) feet for front yards. (c) Front yard fences must be decorative wrought iron or white picket style only. (d) Rear and side yard fences may be wood, vinyl, or wrought iron. (e) Chain link fencing is prohibited in all areas visible from the street."),
        
        ("Section 7.3 - Approved Fence Colors",
         "Fence colors must be pre-approved by the ARC. Standard approved colors include: White, Tan/Beige, Natural wood stain (cedar or redwood tones), Black (wrought iron only), and Forest Green. Any color not listed requires specific ARC approval."),
        
        ("Section 7.4 - Fence Maintenance",
         "Homeowners are responsible for maintaining fences in good condition. Fences must be free of rot, rust, missing boards, or peeling paint. Repairs must be made within 14 days of receiving notice from the HOA. Failure to maintain fences may result in fines starting at $50 per week."),
        
        ("Section 7.5 - Shared Fences",
         "When a fence is located on a property line, both adjacent homeowners share responsibility for maintenance costs. Disputes regarding shared fences should be submitted to the HOA Board for mediation. Neither party may remove or modify a shared fence without written consent from the adjacent neighbor and ARC approval."),
        
        ("Section 7.6 - Variances",
         "Homeowners may request a variance from fence requirements by submitting a written request to the ARC explaining the need for the variance. Variances may be granted for hardship reasons or unique lot configurations. All variances must be approved by a majority vote of the ARC."),
        
        ("Section 7.7 - Application Process",
         "To install a new fence or modify an existing fence: (1) Submit ARC application form with fence plans. (2) Include a site survey showing fence location. (3) Provide material and color specifications. (4) Pay $25 application review fee. (5) Wait for ARC approval before beginning construction. Construction must begin within 90 days of approval."),
        
        ("Section 7.8 - Penalties for Non-Compliance",
         "Fences installed without ARC approval are subject to: First offense - Warning letter and 30 days to submit retroactive application. Second offense - $100 fine and requirement to remove non-compliant fence. Continued non-compliance - Daily fines of $25 and potential legal action. The HOA reserves the right to remove non-compliant structures at the homeowner's expense."),
    ]
    
    for title, text in sections:
        content.append(Paragraph(title, section_style))
        content.append(Paragraph(text, body_style))
        content.append(Spacer(1, 0.1 * inch))
    
    doc.build(content)
    print("Created data/hoa_ccrs.pdf successfully!")

if __name__ == "__main__":
    create_hoa_pdf()
