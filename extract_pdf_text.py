#!/usr/bin/env python3
"""
PDF Text Extraction using Tesseract OCR
Extracts text from all PDF files in a specified folder.
"""

import os
import sys
from pathlib import Path
import argparse
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm


def pdf_to_images(pdf_path, dpi=1000):
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for the conversion (higher = better quality but slower)
    
    Returns:
        List of PIL Image objects
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []


def extract_text_from_image(image, lang='eng'):
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image: PIL Image object
        lang: Language for OCR (default: 'eng' for English)
    
    Returns:
        Extracted text as string
    """
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""


def process_pdf(pdf_path, output_dir=None, lang='eng', dpi=300):
    """
    Process a single PDF file and extract text from all pages.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted text (optional)
        lang: Language for OCR
        dpi: Resolution for PDF to image conversion
    
    Returns:
        Dictionary with filename and extracted text
    """
    pdf_path = Path(pdf_path)
    print(f"\nProcessing: {pdf_path.name}")
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path, dpi=dpi)
    
    if not images:
        return {
            'filename': pdf_path.name,
            'status': 'failed',
            'error': 'Could not convert PDF to images',
            'text': ''
        }
    
    # Extract text from each page
    all_text = []
    for i, image in enumerate(tqdm(images, desc="Extracting text from pages")):
        page_text = extract_text_from_image(image, lang=lang)
        if page_text.strip():
            all_text.append(f"--- Page {i+1} ---\n{page_text}")
    
    combined_text = "\n\n".join(all_text)
    
    # Save to file if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as text file
        text_output_path = output_dir / f"{pdf_path.stem}.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"Text saved to: {text_output_path}")
    
    return {
        'filename': pdf_path.name,
        'status': 'success',
        'pages': len(images),
        'text_length': len(combined_text),
        'text': combined_text
    }


def process_folder(folder_path, output_dir=None, lang='eng', dpi=300, save_json=False):
    """
    Process all PDF files in a folder.
    
    Args:
        folder_path: Path to the folder containing PDF files
        output_dir: Directory to save extracted text
        lang: Language for OCR
        dpi: Resolution for PDF to image conversion
        save_json: Whether to save results as JSON
    
    Returns:
        List of results for all processed PDFs
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return []
    
    # Find all PDF files
    pdf_files = list(folder_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'")
        return []
    
    print(f"Found {len(pdf_files)} PDF file(s) in '{folder_path}'")
    
    # Process each PDF
    results = []
    for pdf_file in pdf_files:
        result = process_pdf(pdf_file, output_dir, lang, dpi)
        results.append(result)
    
    # Save summary as JSON if requested
    if save_json and output_dir:
        output_dir = Path(output_dir)
        json_path = output_dir / f"extraction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create summary without full text for JSON (to keep file size manageable)
        summary = []
        for r in results:
            summary.append({
                'filename': r['filename'],
                'status': r['status'],
                'pages': r.get('pages', 0),
                'text_length': r.get('text_length', 0),
                'error': r.get('error', '')
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary saved to: {json_path}")
    
    return results


def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies_ok = True
    
    # Check for Tesseract executable
    try:
        pytesseract.get_tesseract_version()
        print("✓ Tesseract is installed")
    except Exception:
        print("✗ Tesseract is not installed or not in PATH")
        print("  Install with: brew install tesseract (Mac) or apt-get install tesseract-ocr (Linux)")
        dependencies_ok = False
    
    # Check for poppler (required by pdf2image)
    try:
        from pdf2image import convert_from_path
        # Try a dummy conversion to check if poppler is installed
        print("✓ pdf2image and poppler-utils are available")
    except ImportError:
        print("✗ pdf2image is not installed")
        print("  Install with: pip install pdf2image")
        dependencies_ok = False
    except Exception:
        print("✗ poppler-utils is not installed")
        print("  Install with: brew install poppler (Mac) or apt-get install poppler-utils (Linux)")
        dependencies_ok = False
    
    return dependencies_ok


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files using Tesseract OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/pdfs                     # Process all PDFs in folder
  %(prog)s /path/to/pdfs -o extracted_text   # Save text files to output directory
  %(prog)s /path/to/pdfs -l rus              # Use Russian language for OCR
  %(prog)s /path/to/pdfs --dpi 150           # Use lower DPI for faster processing
  %(prog)s /path/to/single.pdf               # Process a single PDF file
        """
    )
    
    parser.add_argument(
        'input_path',
        help='Path to a folder containing PDFs or a single PDF file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for extracted text files',
        default=None
    )
    
    parser.add_argument(
        '-l', '--lang',
        help='Language for OCR (e.g., eng, rus, deu). Default: eng',
        default='eng'
    )
    
    parser.add_argument(
        '--dpi',
        help='DPI for PDF to image conversion (higher = better quality). Default: 300',
        type=int,
        default=300
    )
    
    parser.add_argument(
        '--json',
        help='Save extraction summary as JSON',
        action='store_true'
    )
    
    parser.add_argument(
        '--check',
        help='Check if all dependencies are installed',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check:
        if check_dependencies():
            print("\nAll dependencies are installed!")
        else:
            print("\nPlease install missing dependencies before running.")
        sys.exit(0)
    
    # Check if dependencies are available
    if not check_dependencies():
        print("\nError: Missing dependencies. Run with --check for details.")
        sys.exit(1)
    
    input_path = Path(args.input_path)
    
    # Process single PDF or folder
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        # Single PDF file
        result = process_pdf(input_path, args.output, args.lang, args.dpi)
        print(f"\nProcessing complete!")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Pages processed: {result['pages']}")
            print(f"Text length: {result['text_length']} characters")
    elif input_path.is_dir():
        # Folder with PDFs
        results = process_folder(input_path, args.output, args.lang, args.dpi, args.json)
        
        # Print summary
        if results:
            print(f"\n{'='*50}")
            print("SUMMARY")
            print(f"{'='*50}")
            success_count = sum(1 for r in results if r['status'] == 'success')
            failed_count = sum(1 for r in results if r['status'] == 'failed')
            print(f"Total PDFs processed: {len(results)}")
            print(f"Successful: {success_count}")
            print(f"Failed: {failed_count}")
            
            if failed_count > 0:
                print("\nFailed files:")
                for r in results:
                    if r['status'] == 'failed':
                        print(f"  - {r['filename']}: {r.get('error', 'Unknown error')}")
    else:
        print(f"Error: '{input_path}' is not a valid PDF file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()