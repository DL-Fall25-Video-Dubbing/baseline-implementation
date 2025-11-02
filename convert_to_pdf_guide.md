# 📄 Guide: Convert Jupyter Notebook to PDF

## ❌ Problem

the notebook `baseline-implementation.ipynb` is giving errors when converting to PDF due to:

1. **Unicode encoding issues** (special characters in the notebook)
2. **Large file size** (2600+ lines, 31 cells)
3. **Missing LaTeX** (required for direct PDF conversion)

---

## ✅ Solutions (Choose One)

### **Solution 1: Print to PDF from Browser (Simplest)**

1. Open the notebook in Jupyter Lab or Jupyter Notebook:

   ```powershell
   jupyter lab "baseline-implementation.ipynb"
   ```

2. In the browser:
   - Click **File** → **Print Preview**
   - Or press `Ctrl + P`
   - Select **Save as PDF** as the printer
   - Click **Save**

**Pros**: No installation needed, preserves formatting
**Cons**: Page breaks might be awkward

---

### **Solution 2: Convert via HTML (Recommended)**

#### Step 1: Convert to HTML with encoding fix

```powershell
python -m nbconvert --to html --output-dir . baseline-implementation.ipynb
```

If encoding error persists, use:

```powershell
python -m nbconvert --to html --no-prompt --output-dir . baseline-implementation.ipynb
```

#### Step 2: Open HTML in browser and print to PDF

1. Open `baseline-implementation.html` in Chrome/Edge
2. Press `Ctrl + P` (Print)
3. Select **Save as PDF**
4. Adjust settings:
   - ✅ Background graphics
   - ✅ Headers and footers (optional)
   - Scale: 90-100%
5. Click **Save**

**Pros**: Clean output, good formatting
**Cons**: Two-step process

---

### **Solution 3: Install LaTeX for Direct PDF (Most Professional)**

#### Step 1: Install MiKTeX (LaTeX for Windows)

Download from: https://miktex.org/download

Or use Chocolatey:

```powershell
choco install miktex
```

#### Step 2: Convert notebook to PDF

```powershell
python -m nbconvert --to pdf baseline-implementation.ipynb
```

**Pros**: Professional PDF with proper typesetting
**Cons**: Large download (~400 MB), takes time to install

---

### **Solution 4: Use Online Converters (with fixes)**

If online converters fail, it's due to encoding issues. Fix it first:

#### Fix Unicode Issues:

1. Open notebook in VS Code
2. Save with encoding: **File** → **Save with Encoding** → **UTF-8**
3. Try these online converters:
   - https://htmtopdf.herokuapp.com/ipynbviewer/ (Notebook → PDF)
   - https://nbviewer.org/ (View online, then print to PDF)
   - https://www.sejda.com/html-to-pdf (After converting to HTML)

**Pros**: No installation
**Cons**: May fail on large files, privacy concerns

---

### **Solution 5: Split Notebook into Sections**

If the notebook is too large, split it:

1. Create separate notebooks for each major section:

   - `01-setup-and-asr.ipynb`
   - `02-translation.ipynb`
   - `03-tts-synthesis.ipynb`
   - `04-lipsync.ipynb`

2. Convert each separately
3. Merge PDFs using:
   - Online: https://www.ilovepdf.com/merge_pdf
   - Tool: Adobe Acrobat, PDFtk

**Pros**: More manageable, better organization
**Cons**: Manual splitting required

---

### **Solution 6: Use Pandoc (Alternative)**

Install Pandoc: https://pandoc.org/installing.html

```powershell
pandoc baseline-implementation.ipynb -o baseline-implementation.pdf
```

**Pros**: Handles many formats
**Cons**: May lose some formatting

---

## 🔧 Quick Fix for Encoding Errors

If you get encoding errors, create a fixed version:

```python
import json
import re

# Read notebook
with open('baseline-implementation.ipynb', 'r', encoding='utf-8', errors='ignore') as f:
    nb = json.load(f)

# Clean cells
for cell in nb['cells']:
    if 'source' in cell:
        # Fix encoding issues
        cell['source'] = [
            line.encode('utf-8', 'ignore').decode('utf-8')
            for line in cell['source']
        ]

# Save cleaned notebook
with open('baseline-implementation-clean.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✓ Created baseline-implementation-clean.ipynb")
```

Then convert the cleaned version:

```powershell
python -m nbconvert --to html baseline-implementation-clean.ipynb
```

---

## 📊 Comparison Table

| Method             | Difficulty    | Quality              | Time   | Requires Install |
| ------------------ | ------------- | -------------------- | ------ | ---------------- |
| Print from Browser | ⭐ Easy       | ⭐⭐⭐ Good          | 2 min  | No               |
| HTML → PDF         | ⭐⭐ Easy     | ⭐⭐⭐⭐ Great       | 5 min  | No               |
| LaTeX (nbconvert)  | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Excellent | 10 min | Yes (MiKTeX)     |
| Online Converter   | ⭐ Easy       | ⭐⭐ Fair            | 5 min  | No               |
| Split Notebook     | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Great       | 30 min | No               |
| Pandoc             | ⭐⭐ Easy     | ⭐⭐⭐ Good          | 5 min  | Yes (Pandoc)     |

---

## 🎯 **Recommended Approach for You:**

**For Quick Result (5 minutes):**

1. Run the encoding fix script below
2. Convert to HTML
3. Print HTML to PDF from browser

**For Professional Result (30 minutes):**

1. Install MiKTeX
2. Run: `python -m nbconvert --to pdf baseline-implementation.ipynb`

---

## 🐛 Common Errors & Fixes

### Error: "UnicodeEncodeError"

**Fix**: Use the encoding fix script above

### Error: "xelatex not found"

**Fix**: Install MiKTeX or use HTML method

### Error: "File too large"

**Fix**: Split notebook into sections or use HTML method

### Error: "Timeout"

**Fix**: Use local conversion, not online

---

## ✅ My Recommendation

**Use Solution 2 (HTML → PDF):**

1. I'll create a cleaned version of the notebook
2. Convert it to HTML
3. You open HTML in browser and print to PDF

This gives you a professional-looking PDF without installing anything!

Would you like me to create the cleaned version now?
