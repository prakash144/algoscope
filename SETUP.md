# 1) Create project folder
mkdir algoscope && cd algoscope

# 2) (Optional but recommended) Use a fresh venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Create the file structure by pasting the code blocks below
#    into the exact paths shown in their headings.

# 4) Install dependencies
pip install -U pip
pip install plotly jinja2 numpy memory-profiler psutil pytest

# 5) Install package in editable mode
pip install -e .

# 6) Run the provided example to generate a report
python examples/compare_linear_vs_binary.py
# or use the provided Quickstart's run_analysis.py workflow
