# Algoscope Setup Guide

## Quick Setup

### 1) Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd algoscope

# Or download and extract the project folder
```

### 2) Create Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3) Install Dependencies

**Option A: Complete Installation (Recommended)**
```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

**Option B: Minimal Installation**
```bash
pip install -U pip
pip install -r requirements-minimal.txt
pip install -e .
```

**Option C: Development Installation**
```bash
pip install -U pip
pip install -r requirements-dev.txt
pip install -e .
```

### 4) Run Examples
```bash
# Basic algorithm comparison
python examples/compare_linear_vs_binary.py

# Sorting algorithms comparison
python examples/sorting_comparison.py

# Google SWE interview prep
python examples/google_swe_interview_prep.py

# Quick interview prep
python examples/quick_interview_prep.py
```

## Project Structure

```
algoscope/
├── src/algoscope/          # Main package source code
│   ├── __init__.py
│   ├── analyze.py          # Core analysis functions
│   ├── complexity.py       # Complexity analysis
│   ├── memory.py          # Memory profiling
│   ├── plotting.py        # Chart generation
│   ├── report.py          # HTML report generation
│   ├── utils.py           # Utility functions
│   └── templates/         # HTML templates
│       └── report.html.j2
├── examples/               # Example scripts and generated reports
│   ├── reports/           # Generated HTML reports
│   ├── json_result/       # Generated JSON data files
│   ├── compare_linear_vs_binary.py
│   ├── sorting_comparison.py
│   ├── google_swe_interview_prep.py
│   └── ... (other examples)
├── tests/                 # Test files
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── requirements-minimal.txt # Minimal dependencies
├── install.sh            # Automated installation script
└── README.md             # Main documentation
```

## Generated Reports

All analysis reports are automatically saved to:
- **HTML Reports**: `examples/reports/`
- **JSON Data**: `examples/json_result/`

## Dependencies

### Core Dependencies
- `jinja2>=3.1.2` - HTML template engine
- `plotly>=5.22.0` - Interactive charts
- `numpy>=1.23.0` - Numerical computing

### Optional Dependencies (for accurate memory profiling)
- `psutil>=5.9.0` - Process and system utilities
- `memory-profiler>=0.61.0` - Memory profiling

### Development Dependencies
- `pytest>=7.0.0` - Testing framework
- `ruff>=0.1.0` - Code linting and formatting

## Troubleshooting

### Memory Profiling Warnings
If you see warnings about `psutil not found`, install it:
```bash
pip install psutil memory-profiler
```

### Import Errors
Make sure you're in the project root directory and have installed the package:
```bash
pip install -e .
```

### Permission Errors
On some systems, you might need to use `python3` instead of `python`:
```bash
python3 examples/compare_linear_vs_binary.py
```

## Next Steps

1. **Explore Examples**: Check out the various example scripts in the `examples/` folder
2. **Read Documentation**: See `README.md` for detailed usage instructions
3. **Create Custom Analysis**: Use the `analyze_functions()` API for your own algorithms
4. **View Reports**: Open the generated HTML files in your browser

## Support

For issues or questions:
- Check the `README.md` for detailed documentation
- Review the example scripts for usage patterns
- Ensure all dependencies are properly installed