# Tips & Tricks

## Convert Jupyter Notebook to Python Script

Since `jupyter nbconvert` may have conflicts, use this Python one-liner:

```bash
python -c "import json; from pathlib import Path; nb = json.load(open('notebooks/01_data_prep.ipynb')); code = '\n\n'.join([''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code' and ''.join(cell['source']).strip()]); Path('scripts').mkdir(exist_ok=True); Path('scripts/01_data_prep.py').write_text(code); print('Converted to scripts/01_data_prep.py')"
```

**What it does:**
- Reads the notebook JSON
- Extracts only code cells (skips markdown)
- Writes to `scripts/` directory
- Creates the output file with `.py` extension

**Note:** Remove `%matplotlib inline` and other Jupyter magic commands from the converted file if needed.

