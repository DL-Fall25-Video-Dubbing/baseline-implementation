"""
Fix Notebook Encoding Issues and Convert to HTML/PDF

This script:
1. Cleans Unicode encoding issues from the notebook
2. Converts to HTML
3. Optionally converts to PDF if LaTeX is available
"""

import json
import subprocess
import sys
from pathlib import Path

def clean_notebook(input_path, output_path):
    """Clean encoding issues from notebook"""
    print(f"Reading notebook: {input_path}")
    
    # Read notebook with error handling
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        nb = json.load(f)
    
    # Clean each cell
    cells_cleaned = 0
    for cell in nb['cells']:
        if 'source' in cell:
            original_source = cell['source']
            
            # Clean each line
            cleaned_source = []
            for line in original_source:
                # Remove problematic characters
                try:
                    # Try to encode/decode to catch issues
                    cleaned_line = line.encode('utf-8', 'ignore').decode('utf-8')
                    # Remove surrogate characters
                    cleaned_line = ''.join(char for char in cleaned_line 
                                          if not (0xD800 <= ord(char) <= 0xDFFF))
                    cleaned_source.append(cleaned_line)
                except:
                    # If still fails, skip the line or use ASCII
                    cleaned_line = line.encode('ascii', 'ignore').decode('ascii')
                    cleaned_source.append(cleaned_line)
            
            if cleaned_source != original_source:
                cells_cleaned += 1
                cell['source'] = cleaned_source
    
    # Save cleaned notebook
    print(f"Writing cleaned notebook: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"✓ Cleaned {cells_cleaned} cells")
    return output_path

def convert_to_html(notebook_path):
    """Convert notebook to HTML"""
    print(f"\nConverting to HTML...")
    
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'html',
        '--no-prompt',  # Remove input prompts
        str(notebook_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ HTML conversion successful")
        
        html_path = notebook_path.with_suffix('.html')
        print(f"✓ Created: {html_path}")
        print(f"\nTo convert to PDF:")
        print(f"  1. Open {html_path.name} in Chrome/Edge")
        print(f"  2. Press Ctrl+P")
        print(f"  3. Select 'Save as PDF'")
        print(f"  4. Save")
        
        return html_path
    except subprocess.CalledProcessError as e:
        print(f"✗ HTML conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def convert_to_pdf(notebook_path):
    """Convert notebook to PDF (requires LaTeX)"""
    print(f"\nAttempting PDF conversion...")
    
    # Check if pdflatex is available
    try:
        subprocess.run(['pdflatex', '--version'], 
                      capture_output=True, check=True)
        print("✓ LaTeX found")
    except:
        print("✗ LaTeX not found - PDF conversion requires MiKTeX/TeXLive")
        print("  Install from: https://miktex.org/download")
        return None
    
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'pdf',
        str(notebook_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ PDF conversion successful")
        
        pdf_path = notebook_path.with_suffix('.pdf')
        print(f"✓ Created: {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"✗ PDF conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def main():
    # Configuration
    input_notebook = Path("baseline-implementation.ipynb")
    cleaned_notebook = Path("baseline-implementation-clean.ipynb")
    
    if not input_notebook.exists():
        print(f"✗ Error: {input_notebook} not found")
        print(f"  Current directory: {Path.cwd()}")
        return
    
    print("="*60)
    print("Notebook to PDF Converter")
    print("="*60)
    
    # Step 1: Clean notebook
    try:
        clean_notebook(input_notebook, cleaned_notebook)
    except Exception as e:
        print(f"✗ Error cleaning notebook: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Convert to HTML
    html_path = convert_to_html(cleaned_notebook)
    
    # Step 3: Try PDF (optional)
    print("\n" + "="*60)
    pdf_path = convert_to_pdf(cleaned_notebook)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"✓ Original notebook: {input_notebook}")
    print(f"✓ Cleaned notebook: {cleaned_notebook}")
    if html_path:
        print(f"✓ HTML output: {html_path}")
        print(f"\n📄 Next step: Open {html_path.name} and print to PDF")
    if pdf_path:
        print(f"✓ PDF output: {pdf_path}")
    
    print("\n✓ Conversion complete!")

if __name__ == "__main__":
    main()
