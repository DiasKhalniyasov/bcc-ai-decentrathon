# BCC Contract Processing Pipeline

## Prerequisites
- Install Tesseract OCR: `brew install tesseract` (Mac) or `apt-get install tesseract-ocr` (Linux)
- Install Poppler: `brew install poppler` (Mac) or `apt-get install poppler-utils` (Linux)
- Install Python dependencies: `pip install -r requirements_ocr.txt`
- Ensure Ollama is running with gemma3:27b model

## Pipeline 1: Baseline OCR Processing

### Step 1: Extract text from PDFs using OCR
```bash
python /Users/diaskhalniyasov/Desktop/work/bcc/extract_pdf_text.py data/ -o predictions_baseline -l rus+eng
```
- Processes all PDFs in `data/` folder
- Outputs OCR text files to `predictions_baseline/`
- Uses Russian and English language models for OCR

### Step 2: Process OCR output with LLM
```bash
python ollama_qwen3_predictor.py predictions_baseline/
```
- Processes all .txt files (excluding _ocr suffix and _pages directories)
- Extracts contract fields using gemma3:27b model
- Saves predictions to `predictions/` directory

### Step 3: Evaluate predictions
```bash
python evaluate_predictions.py --gt-dir gt_answers --pred-dir predictions
```
- Compares predictions against ground truth
- Outputs evaluation metrics

## Pipeline 2: Dots Feature Processing

### Step 1: Gather all dots JSONs for a specific feature
```bash
python gather_jsons.py C781 C781.json
```
- Collects all JSON files for feature C781
- Outputs consolidated JSON file

### Step 2: Process dots predictions
```bash
python ollama_qwen3_predictor.py predictions_dots/
```
- Automatically detects 'dots' in directory name
- Processes all .json files in the directory
- Extracts contract fields using gemma3:27b model
- Saves predictions to `predictions/` directory

### Step 3: Evaluate predictions
```bash
python evaluate_predictions.py --gt-dir gt_answers --pred-dir predictions
```
- Compares predictions against ground truth
- Outputs evaluation metrics

## Directory Structure
```
bcc/
├── data/                    # Input PDF files
├── predictions_baseline/    # OCR output text files
├── predictions_dots/        # Dots feature JSON files
├── predictions/            # Final predictions (JSON)
├── gt_answers/             # Ground truth answers
└── evaluation_results.json # Evaluation metrics
```

## Notes
- The ollama_qwen3_predictor.py automatically handles different input types:
  - For directories with 'dots' in name: processes .json files
  - For regular directories: processes .txt files (excluding _ocr and _pages)
- Ensure Ollama is running before processing: `ollama serve`
- Check model availability: `ollama list`