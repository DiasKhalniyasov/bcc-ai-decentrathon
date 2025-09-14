#!/usr/bin/env python3
"""
JSON Predictions Evaluation Script
Compares model predictions against ground truth answers and calculates various metrics.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import Levenshtein
import jiwer
import argparse
from datetime import datetime


class MetricsCalculator:
    """Calculate various metrics for JSON predictions evaluation."""
    
    def __init__(self, gt_dir: str = "gt_answers", pred_dir: str = "predictions"):
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.results = defaultdict(list)
        self.field_metrics = defaultdict(lambda: defaultdict(list))
        
    def load_json_file(self, filepath: Path) -> Tuple[Optional[Dict], bool]:
        """Load and validate JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, True
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filepath}")
            return None, False
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None, False
            
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate using jiwer."""
        if not reference:
            return 0.0 if not hypothesis else 1.0
        if not hypothesis:
            return 1.0
        
        # jiwer's cer function works on character level
        cer_value = jiwer.cer(reference, hypothesis)
        return cer_value
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate using jiwer."""
        if not reference:
            return 0.0 if not hypothesis else 1.0
        if not hypothesis:
            return 1.0
            
        # jiwer's wer function works on word level
        wer_value = jiwer.wer(reference, hypothesis)
        return wer_value
    
    def normalized_levenshtein(self, reference: str, hypothesis: str) -> float:
        """Calculate Normalized Levenshtein Distance."""
        if not reference and not hypothesis:
            return 0.0
        max_len = max(len(reference), len(hypothesis))
        if max_len == 0:
            return 0.0
        distance = Levenshtein.distance(reference, hypothesis)
        return distance / max_len
    
    def calculate_field_metrics(self, gt_value: Any, pred_value: Any, field_name: str) -> Dict[str, float]:
        """Calculate metrics for a single field."""
        metrics = {}
        
        # Convert to string for text-based metrics
        gt_str = str(gt_value) if gt_value is not None else ""
        pred_str = str(pred_value) if pred_value is not None else ""
        
        # Exact match (accuracy)
        metrics['exact_match'] = 1.0 if gt_value == pred_value else 0.0
        
        # Text-based metrics
        metrics['cer'] = self.calculate_cer(gt_str, pred_str)
        metrics['wer'] = self.calculate_wer(gt_str, pred_str)
        metrics['normalized_levenshtein'] = self.normalized_levenshtein(gt_str, pred_str)
        
        # For F1 calculation (treating as binary classification per field)
        metrics['true_positive'] = 1 if gt_value == pred_value and gt_value is not None else 0
        metrics['false_positive'] = 1 if gt_value != pred_value and pred_value is not None else 0
        metrics['false_negative'] = 1 if gt_value != pred_value and gt_value is not None else 0
        metrics['true_negative'] = 1 if gt_value == pred_value and gt_value is None else 0
        
        return metrics
    
    def calculate_f1_score(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate_document(self, gt_data: Dict, pred_data: Dict, filename: str) -> Dict:
        """Evaluate a single document pair."""
        doc_results = {
            'filename': filename,
            'fields': {},
            'document_exact_match': False,
            'json_valid': True,  # Already validated if we got here
            'schema_consistent': True
        }
        
        # Check schema consistency (all required keys present)
        gt_keys = set(gt_data.keys())
        pred_keys = set(pred_data.keys())
        doc_results['schema_consistent'] = gt_keys == pred_keys
        doc_results['missing_keys'] = list(gt_keys - pred_keys)
        doc_results['extra_keys'] = list(pred_keys - gt_keys)
        
        # Evaluate each field
        all_exact_match = True
        for field in gt_keys:
            gt_value = gt_data.get(field)
            pred_value = pred_data.get(field)
            
            field_metrics = self.calculate_field_metrics(gt_value, pred_value, field)
            doc_results['fields'][field] = field_metrics
            
            # Store for aggregate calculations
            for metric_name, metric_value in field_metrics.items():
                self.field_metrics[field][metric_name].append(metric_value)
            
            if field_metrics['exact_match'] == 0:
                all_exact_match = False
        
        doc_results['document_exact_match'] = all_exact_match
        return doc_results
    
    def run_evaluation(self) -> Dict:
        """Run evaluation on all documents."""
        overall_results = {
            'timestamp': datetime.now().isoformat(),
            'documents': [],
            'aggregate_metrics': {},
            'field_level_metrics': {},
            'summary': {}
        }
        
        # Get all GT files
        gt_files = sorted(self.gt_dir.glob("*.json"))
        total_documents = len(gt_files)
        valid_json_count = 0
        schema_consistent_count = 0
        exact_match_count = 0
        
        for gt_file in gt_files:
            filename = gt_file.name
            pred_file = self.pred_dir / filename
            
            # Load files
            gt_data, gt_valid = self.load_json_file(gt_file)
            pred_data, pred_valid = self.load_json_file(pred_file)
            
            if not gt_valid:
                print(f"Skipping {filename}: Invalid GT JSON")
                continue
                
            if not pred_valid:
                # Count as invalid prediction
                doc_result = {
                    'filename': filename,
                    'json_valid': False,
                    'error': 'Invalid or missing prediction JSON'
                }
                overall_results['documents'].append(doc_result)
                continue
            
            valid_json_count += 1
            
            # Evaluate document
            doc_result = self.evaluate_document(gt_data, pred_data, filename)
            overall_results['documents'].append(doc_result)
            
            if doc_result['schema_consistent']:
                schema_consistent_count += 1
            if doc_result['document_exact_match']:
                exact_match_count += 1
        
        # Calculate aggregate metrics
        overall_results['summary'] = {
            'total_documents': total_documents,
            'json_validity_rate': valid_json_count / total_documents if total_documents > 0 else 0,
            'schema_consistency_rate': schema_consistent_count / total_documents if total_documents > 0 else 0,
            'exact_match_rate': exact_match_count / total_documents if total_documents > 0 else 0
        }
        
        # Calculate field-level aggregate metrics
        for field, metrics in self.field_metrics.items():
            field_summary = {}
            
            # Calculate averages
            for metric_name in ['cer', 'wer', 'normalized_levenshtein', 'exact_match']:
                if metric_name in metrics:
                    field_summary[f'avg_{metric_name}'] = np.mean(metrics[metric_name])
                    field_summary[f'std_{metric_name}'] = np.std(metrics[metric_name])
            
            # Calculate F1 scores
            tp = sum(metrics.get('true_positive', []))
            fp = sum(metrics.get('false_positive', []))
            fn = sum(metrics.get('false_negative', []))
            
            f1_metrics = self.calculate_f1_score(tp, fp, fn)
            field_summary.update(f1_metrics)
            
            overall_results['field_level_metrics'][field] = field_summary
        
        # Calculate overall averages across all fields
        all_cer = []
        all_wer = []
        all_leven = []
        all_exact = []
        
        for field_metrics in self.field_metrics.values():
            all_cer.extend(field_metrics.get('cer', []))
            all_wer.extend(field_metrics.get('wer', []))
            all_leven.extend(field_metrics.get('normalized_levenshtein', []))
            all_exact.extend(field_metrics.get('exact_match', []))
        
        overall_results['aggregate_metrics'] = {
            'avg_cer': np.mean(all_cer) if all_cer else 0,
            'avg_wer': np.mean(all_wer) if all_wer else 0,
            'avg_normalized_levenshtein': np.mean(all_leven) if all_leven else 0,
            'avg_field_accuracy': np.mean(all_exact) if all_exact else 0
        }
        
        return overall_results
    
    def save_results(self, results: Dict, output_dir: str = "."):
        """Save evaluation results to files."""
        output_dir = Path(output_dir)
        
        # Create simplified results for JSON (without per-field metrics in documents)
        simplified_results = {
            'timestamp': results['timestamp'],
            'documents': [],
            'aggregate_metrics': results['aggregate_metrics'],
            'field_level_metrics': results['field_level_metrics'],
            'summary': results['summary']
        }
        
        # Add simplified document info (without per-field metrics)
        for doc in results['documents']:
            simplified_doc = {
                'filename': doc['filename'],
                'document_exact_match': doc.get('document_exact_match', False),
                'json_valid': doc.get('json_valid', False),
                'schema_consistent': doc.get('schema_consistent', False),
                'missing_keys': doc.get('missing_keys', []),
                'extra_keys': doc.get('extra_keys', [])
            }
            if 'error' in doc:
                simplified_doc['error'] = doc['error']
            simplified_results['documents'].append(simplified_doc)
        
        # Save simplified JSON results
        json_output = output_dir / "evaluation_results.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)
        print(f"Full results saved to {json_output}")
        
        # Create summary DataFrame for CSV
        summary_data = []
        
        # Document-level summary
        for doc in results['documents']:
            if doc.get('json_valid', False):
                row = {
                    'filename': doc['filename'],
                    'exact_match': doc.get('document_exact_match', False),
                    'schema_consistent': doc.get('schema_consistent', False)
                }
                # Add average metrics per document
                if 'fields' in doc:
                    doc_cer = []
                    doc_wer = []
                    doc_leven = []
                    for field_metrics in doc['fields'].values():
                        doc_cer.append(field_metrics['cer'])
                        doc_wer.append(field_metrics['wer'])
                        doc_leven.append(field_metrics['normalized_levenshtein'])
                    row['avg_cer'] = np.mean(doc_cer)
                    row['avg_wer'] = np.mean(doc_wer)
                    row['avg_normalized_levenshtein'] = np.mean(doc_leven)
                summary_data.append(row)
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            csv_output = output_dir / "evaluation_summary.csv"
            df_summary.to_csv(csv_output, index=False)
            print(f"Summary CSV saved to {csv_output}")
        
        # Create field-level metrics DataFrame
        field_data = []
        for field, metrics in results['field_level_metrics'].items():
            row = {'field': field}
            row.update(metrics)
            field_data.append(row)
        
        if field_data:
            df_fields = pd.DataFrame(field_data)
            field_csv = output_dir / "field_metrics.csv"
            df_fields.to_csv(field_csv, index=False)
            print(f"Field metrics CSV saved to {field_csv}")
        
        # Print summary to console
        self.print_summary(results)
    
    def print_summary(self, results: Dict):
        """Print evaluation summary to console."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        summary = results['summary']
        print(f"\nDocument-Level Metrics:")
        print(f"  Total Documents: {summary['total_documents']}")
        print(f"  JSON Validity Rate: {summary['json_validity_rate']:.2%}")
        print(f"  Schema Consistency Rate: {summary['schema_consistency_rate']:.2%}")
        print(f"  Exact Match Rate: {summary['exact_match_rate']:.2%}")
        
        agg = results['aggregate_metrics']
        print(f"\nAggregate Metrics (All Fields):")
        print(f"  Average CER: {agg['avg_cer']:.4f}")
        print(f"  Average WER: {agg['avg_wer']:.4f}")
        print(f"  Average Normalized Levenshtein: {agg['avg_normalized_levenshtein']:.4f}")
        print(f"  Average Field Accuracy: {agg['avg_field_accuracy']:.2%}")
        
        print(f"\nField-Level Metrics:")
        for field, metrics in results['field_level_metrics'].items():
            print(f"\n  {field}:")
            print(f"    Accuracy: {metrics.get('avg_exact_match', 0):.2%}")
            print(f"    CER: {metrics.get('avg_cer', 0):.4f}")
            print(f"    WER: {metrics.get('avg_wer', 0):.4f}")
            print(f"    F1 Score: {metrics.get('f1_score', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate JSON predictions against ground truth')
    parser.add_argument('--gt-dir', default='gt_answers', help='Directory with ground truth JSON files')
    parser.add_argument('--pred-dir', default='predictions', help='Directory with prediction JSON files')
    parser.add_argument('--output-dir', default='.', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory '{args.gt_dir}' not found")
        sys.exit(1)
    if not os.path.exists(args.pred_dir):
        print(f"Error: Predictions directory '{args.pred_dir}' not found")
        sys.exit(1)
    
    # Run evaluation
    calculator = MetricsCalculator(args.gt_dir, args.pred_dir)
    results = calculator.run_evaluation()
    calculator.save_results(results, args.output_dir)
    
    print(f"\nEvaluation complete! Check output files for detailed results.")


if __name__ == "__main__":
    main()