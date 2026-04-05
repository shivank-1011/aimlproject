from fpdf import FPDF
import datetime

class StudyPlanPDF(FPDF):
    def header(self):
        # Logo or Title
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Student Performance Diagnosis & Study Plan', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
        self.ln(10)

    def footer(self):
        # Page numbering
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.ln(2)
        self.set_font('Arial', '', 10)
        # Using multi_cell to handle line breaks and long text
        self.multi_cell(0, 6, content.strip())
        self.ln(10)

def generate_study_plan_pdf(student_name, report_text, output_path=None):
    """
    Parses the agent's report text and generates a formatted PDF.
    Expects specific sections (1. Learning Diagnosis, etc.)
    """
    # Simple sanitization to replace non-latin-1 characters with equivalents
    def sanitize_text(text):
        # Replace common issue-causing characters
        replacements = {
            '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'", 
            '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
            '\u2022': '*', # bullet point
            '🤖': 'AI', '🎯': 'Goal', '🚨': 'Critical', '🌟': 'Star', # Emojis
        }
        for search, replace in replacements.items():
            text = text.replace(search, replace)
        # Fallback for remaining non-latin-1
        return text.encode('ascii', 'ignore').decode('ascii')

    report_text = sanitize_text(report_text)
    
    pdf = StudyPlanPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Student: {student_name}', 0, 1, 'L')
    pdf.ln(5)

    # Split the report into sections (crude parsing based on numbers or typical sections)
    # The agent uses numbered sections 1-5
    sections = report_text.split("\n")
    current_title = ""
    current_content = ""
    
    for line in sections:
        line = line.strip()
        if not line: continue
        
        # Check if the line is a section header (e.g. "1. Learning Diagnosis")
        if any(line.startswith(str(i) + ".") for i in range(1, 10)) or \
           any(line.upper().startswith(s) for s in ["STUDY PLAN", "WEEKLY GOALS", "RESOURCES", "NEXT STEPS"]):
            
            if current_title:
                pdf.add_section(current_title, current_content)
                
            current_title = line
            current_content = ""
        else:
            current_content += line + "\n"
            
    # Add final section
    if current_title:
        pdf.add_section(current_title, current_content)

    if output_path:
        pdf.output(output_path)
    else:
        return pdf.output(dest='S').encode('latin-1')

if __name__ == "__main__":
    # Test script
    test_text = """
    1. Learning Diagnosis:
    Student Ayush shows a critical gap in AIML with a score of 32. 
    However, they excel in FSD with 85.
    
    2. Study Plan:
    Focus during weeks 1-2 on core mathematics and AI basics.
    
    3. Weekly Goals:
    - Master backpropagation.
    - Build a small linear regression model.
    """
    generate_study_plan_pdf("Ayush", test_text, "test_report.pdf")
    print("Test PDF generated.")
