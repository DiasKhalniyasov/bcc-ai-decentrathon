# JSON Predictions Evaluation Metrics Documentation

## Overview

This document describes the evaluation metrics used to assess the quality of JSON predictions compared to ground truth answers. The evaluation system is designed for structured document extraction tasks where the model outputs JSON with specific fields.

## Metrics Description

### 1. Character Error Rate (CER)

**Definition:** The ratio of character-level edits (insertions, deletions, substitutions) needed to transform the predicted text into the ground truth text.

**Formula:**
```
CER = Levenshtein_distance(prediction, ground_truth) / len(ground_truth)
```

**Range:** [0, ∞) where 0 is perfect match

**Interpretation:**
- CER = 0: Perfect character-level match
- CER < 0.1: Very good accuracy
- CER < 0.3: Acceptable accuracy
- CER > 0.5: Poor accuracy

**Example:**
- Ground Truth: "24022311"
- Prediction: "124022311"
- CER = 1/8 = 0.125 (one extra character)

### 2. Word Error Rate (WER)

**Definition:** The ratio of word-level edits needed to transform the predicted text into the ground truth text.

**Formula:**
```
WER = Word_Levenshtein_distance(prediction, ground_truth) / num_words(ground_truth)
```

**Range:** [0, ∞) where 0 is perfect match

**Interpretation:**
- WER = 0: Perfect word-level match
- WER < 0.1: Very good accuracy
- WER < 0.3: Acceptable accuracy
- WER > 0.5: Poor accuracy

**Example:**
- Ground Truth: "ОАО БМЗ управляющая компания"
- Prediction: "ОАО БМЗ компания"
- WER = 1/4 = 0.25 (one word missing)

### 3. Normalized Levenshtein Distance

**Definition:** The Levenshtein distance normalized by the maximum length of the two strings, providing a metric bounded between 0 and 1.

**Formula:**
```
Normalized_Levenshtein = Levenshtein_distance(prediction, ground_truth) / max(len(prediction), len(ground_truth))
```

**Range:** [0, 1] where 0 is perfect match and 1 is completely different

**Interpretation:**
- 0.0: Identical strings
- < 0.1: Very similar
- < 0.3: Moderately similar
- > 0.5: Very different

### 4. Field-level Accuracy

**Definition:** The percentage of fields that exactly match between prediction and ground truth.

**Formula:**
```
Field_Accuracy = (Number of exactly matching fields) / (Total number of fields)
```

**Range:** [0, 1] where 1 is perfect accuracy

**Interpretation:**
- 1.0: All fields match exactly
- > 0.8: Good field extraction
- > 0.6: Acceptable extraction
- < 0.5: Poor extraction

### 5. F1-Score per Field

**Definition:** The harmonic mean of precision and recall for each field, treating field extraction as a binary classification problem.

**Components:**
- **Precision:** Of all predicted non-null values, how many are correct?
- **Recall:** Of all ground truth non-null values, how many were correctly predicted?
- **F1:** Harmonic mean of precision and recall

**Formula:**
```
Precision = True_Positives / (True_Positives + False_Positives)
Recall = True_Positives / (True_Positives + False_Negatives)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Range:** [0, 1] where 1 is perfect score

**Interpretation:**
- F1 > 0.9: Excellent field extraction
- F1 > 0.7: Good extraction
- F1 > 0.5: Acceptable
- F1 < 0.5: Poor extraction

### 6. Exact Match per Document

**Definition:** The percentage of documents where ALL fields exactly match the ground truth.

**Formula:**
```
Exact_Match_Rate = (Number of perfectly matched documents) / (Total number of documents)
```

**Range:** [0, 1] where 1 means all documents are perfect

**Interpretation:**
- This is the strictest metric
- Even one wrong field in a document counts as a failure
- Useful for understanding overall system reliability

### 7. JSON Validity

**Definition:** The percentage of predictions that are valid JSON format.

**Formula:**
```
JSON_Validity_Rate = (Number of valid JSON files) / (Total number of prediction files)
```

**Range:** [0, 1] where 1 means all outputs are valid JSON

**Interpretation:**
- Should ideally be 1.0 (100%)
- < 1.0 indicates parsing or format errors in model output

### 8. Schema Consistency

**Definition:** The percentage of JSON predictions that contain all required keys from the ground truth schema.

**Formula:**
```
Schema_Consistency_Rate = (Number of predictions with correct schema) / (Total number of predictions)
```

**Range:** [0, 1] where 1 means all predictions have the correct structure

**Interpretation:**
- 1.0: Model consistently outputs all required fields
- < 1.0: Model sometimes misses or adds unexpected fields

## Usage Instructions

### Basic Usage

Run the evaluation script with default directories:
```bash
python evaluate_predictions.py
```

### Custom Directories

Specify custom directories for ground truth and predictions:
```bash
python evaluate_predictions.py --gt-dir path/to/ground_truth --pred-dir path/to/predictions --output-dir path/to/results
```

### Input Format

Both ground truth and predictions should be JSON files with the same naming convention:
- Ground truth: `gt_answers/document_id.json`
- Predictions: `predictions/document_id.json`

Example JSON structure:
```json
{
  "номер_контракта": "24022311",
  "дата_контракта": "2024-04-12",
  "дата_окончания_контракта": "2026-06-30",
  "наименование_контрагента": "ОАО БМЗ",
  "страна_контрагента": "BY.Беларусь",
  "наименование_банка_контрагента": null,
  "сумма_контракта": 100000000.00,
  "валюта_контракта": "RUB",
  "валюта_платежа": "RUB"
}
```

### Output Files

The script generates three output files:

1. **evaluation_results.json**: Complete detailed results including:
   - Per-document metrics
   - Per-field metrics
   - Aggregate statistics
   - Error details

2. **evaluation_summary.csv**: Document-level summary with:
   - Document names
   - Exact match status
   - Average CER, WER, Levenshtein per document
   - Schema consistency

3. **field_metrics.csv**: Field-level aggregate metrics:
   - Average accuracy, CER, WER per field
   - F1-scores per field
   - Standard deviations

## Interpreting Results

### Good Performance Indicators
- JSON Validity Rate = 100%
- Schema Consistency Rate = 100%
- Average CER < 0.1
- Average WER < 0.15
- Field Accuracy > 85%
- F1-scores > 0.8 for most fields

### Common Issues and Solutions

1. **High CER/WER but good Field Accuracy**
   - Minor OCR errors or formatting differences
   - Check for systematic patterns (extra spaces, punctuation)

2. **Low Schema Consistency**
   - Model not outputting all required fields
   - Check model prompts and training data

3. **Low F1-score for specific fields**
   - Model struggling with particular field types
   - May need field-specific improvements

4. **Low Exact Match Rate but good individual metrics**
   - Model making small errors across many documents
   - Focus on the most common error patterns

## Best Practices

1. **Regular Evaluation**: Run evaluation after each model update
2. **Track Trends**: Monitor metrics over time to ensure improvements
3. **Field Analysis**: Pay attention to consistently problematic fields
4. **Error Analysis**: Review documents with lowest scores to identify patterns
5. **Balanced Metrics**: Consider multiple metrics, not just one
6. **Threshold Setting**: Define acceptable thresholds for your use case

## Advanced Analysis

### Weighted Metrics
For fields with different importance levels, consider implementing weighted averages:
```python
weights = {
    "номер_контракта": 2.0,  # Critical field
    "сумма_контракта": 1.5,  # Important field
    "наименование_банка_контрагента": 0.5  # Optional field
}
```

### Confidence Thresholds
If your model provides confidence scores, correlate them with accuracy metrics to find optimal thresholds.

### Error Categories
Classify errors into categories:
- Format errors (date formats, number formats)
- OCR errors (character substitutions)
- Semantic errors (wrong field extraction)
- Structural errors (missing/extra fields)

## Troubleshooting

### Common Errors

1. **"File not found" errors**
   - Check that file names match exactly between gt_answers/ and predictions/
   - Ensure directories exist and contain JSON files

2. **JSON decode errors**
   - Validate JSON syntax using a JSON validator
   - Check for encoding issues (use UTF-8)

3. **Key errors**
   - Ensure all required fields are present in both files
   - Handle null values appropriately

4. **Memory issues with large datasets**
   - Process files in batches
   - Use generator patterns for large-scale evaluation

## References

- Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals"
- Word Error Rate in ASR systems: https://en.wikipedia.org/wiki/Word_error_rate
- F1 Score: https://en.wikipedia.org/wiki/F-score