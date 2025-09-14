#!/usr/bin/env python3
"""
Ollama Qwen3 Predictor Script
Loads Qwen3 model from Ollama and runs predictions on full text from OCR output.
Supports both plain text files and JSON array structures with text fields.
Uses structured output with Pydantic models for reliable JSON extraction.
Implements text preprocessing to extract relevant sections for each field.
"""

import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Union
import argparse
import subprocess
from pydantic import BaseModel, Field
from ollama import chat
import re


class ContractData(BaseModel):
    """Pydantic model for contract data structure."""
    номер_контракта: Optional[str] = Field(None, description="Contract number")
    дата_контракта: Optional[str] = Field(None, description="Contract date in YYYY-MM-DD format")
    дата_окончания_контракта: Optional[str] = Field(None, description="Contract end date in YYYY-MM-DD format")
    наименование_контрагента: Optional[str] = Field(None, description="Counterparty name")
    страна_контрагента: Optional[str] = Field(None, description="Counterparty country")
    сумма_контракта: Optional[float] = Field(None, description="Contract amount")
    валюта_контракта: Optional[str] = Field(None, description="Contract currency")
    валюта_платежа: Optional[str] = Field(None, description="Payment currency")


class OllamaQwen3Predictor:
    """Handle predictions using Qwen3 model via Ollama."""
    
    def __init__(self, model_name: str = "gemma3:27b"):
        self.model_name = model_name
        
    def check_model_available(self) -> bool:
        """Check if the model is available in Ollama."""
        try:
            # Try to list models using ollama command
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, 
                                  text=True, 
                                  check=False)
            if result.returncode == 0:
                model_base = self.model_name.split(':')[0]
                return model_base in result.stdout or 'qwen' in result.stdout.lower()
            return False
        except Exception:
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not available."""
        print(f"Pulling {self.model_name} model...")
        try:
            subprocess.run(['ollama', 'pull', self.model_name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error pulling model: {e}")
            return False
    
    def extract_text_from_json_array(self, json_data: Union[List, Dict]) -> str:
        """Extract and combine ALL text from JSON array structure."""
        combined_text = []
        
        if isinstance(json_data, list):
            # Sort by page position if bbox is available
            sorted_items = json_data
            if all('bbox' in item for item in json_data if isinstance(item, dict)):
                sorted_items = sorted(json_data, key=lambda x: (
                    x.get('bbox', [0, 0])[1],  # y-coordinate (top)
                    x.get('bbox', [0, 0])[0]   # x-coordinate (left)
                ))
            
            for item in sorted_items:
                if isinstance(item, dict) and 'text' in item:
                    text = item['text'].strip()
                    if text:
                        # Clean up markdown-like headers
                        if text.startswith('##'):
                            text = text.replace('##', '').strip()
                        combined_text.append(text)
        
        elif isinstance(json_data, dict):
            # If it's a dict, look for text fields recursively
            for value in json_data.values():
                if isinstance(value, str):
                    combined_text.append(value)
                elif isinstance(value, (list, dict)):
                    nested_text = self.extract_text_from_json_array(value)
                    if nested_text:
                        combined_text.append(nested_text)
        
        return '\n'.join(combined_text)
    
    def filter_json_array_by_pattern(self, json_data: List[Dict], pattern: str) -> str:
        """Filter JSON array items by regex pattern and combine matching texts."""
        if not pattern:
            return ""
        
        matching_texts = []
        
        for item in json_data:
            if isinstance(item, dict) and 'text' in item:
                text = item['text'].strip()
                # Check if this text matches the pattern
                if re.search(pattern, text, re.IGNORECASE):
                    # Clean up markdown-like headers
                    if text.startswith('##'):
                        text = text.replace('##', '').strip()
                    matching_texts.append(text)
        
        return '\n'.join(matching_texts)
    
    def extract_relevant_sections_from_json(self, json_data: List[Dict]) -> Dict[str, str]:
        """Extract relevant text sections for each field from JSON array."""
        sections = {}
        
        # Define patterns for each field
        patterns = {
            'номер_контракта': r'№',  # Text must contain №
            'дата_контракта': None,  # Use header - first few items
            'дата_окончания_контракта': r'действ\w*',  # Matches "действует", "действуют", etc.
            'наименование_контрагента': r'именуем\w*',  # Matches "именуемый", "именуемое", etc.
            'страна_контрагента': r'именуем\w*',  # Same as counterparty
            'сумма_контракта': r'(?:стоимост\w*|сумм\w*)',  # Matches "стоимость", "сумма", etc.
            'валюта_контракта': r'(?:стоимост\w*|сумм\w*|валют\w*)',  # Same as amount + "валюта", etc.
            'валюта_платежа': r'(?:валют\w*.*платеж\w*|платеж\w*.*валют\w*)'  # Matches combinations of "валюта" and "платеж"
        }
        
        # Process each field
        for field, pattern in patterns.items():
            if field == 'дата_контракта':
                # For contract date, use header (first few items with text)
                header_texts = []
                for item in json_data[:5]:  # First 5 items
                    if isinstance(item, dict) and 'text' in item:
                        text = item['text'].strip()
                        if text:
                            if text.startswith('##'):
                                text = text.replace('##', '').strip()
                            header_texts.append(text)
                sections[field] = '\n'.join(header_texts)
            
            elif field == 'страна_контрагента':
                # Use same section as counterparty
                sections[field] = sections.get('наименование_контрагента', '')
            
            elif field == 'валюта_контракта':
                # Use same section as amount
                sections[field] = sections.get('сумма_контракта', '')
            
            elif pattern:
                # Filter by pattern
                sections[field] = self.filter_json_array_by_pattern(json_data, pattern)
            else:
                sections[field] = ''
        
        return sections
    
    def create_field_specific_prompt(self, field: str, text_section: str) -> str:
        """Create a specialized prompt for each field type."""
        prompts = {
            'номер_контракта': f"""Extract ONLY the contract number from this text.
Look for patterns like "ДОГОВОР №", "№", "контракт №".
Return just the number/identifier, nothing else.
If data not found return null.

Text: {text_section}

Contract number:""",
            
            'дата_контракта': f"""Extract the contract signing date from this text.
Look for patterns like "от" followed by a date.
Convert to YYYY-MM-DD format.
If data not found return null.

Text: {text_section}

Date (YYYY-MM-DD):""",
            
            'дата_окончания_контракта': f"""Extract the contract end/expiration date.
Convert to YYYY-MM-DD format.
If data not found return null.

Text: {text_section}

End date (YYYY-MM-DD):""",
            
            'наименование_контрагента': f"""Extract the counterparty company name. Counterparty is company NOT from Kazakhstan.
            Usually company names have 'ТОО', 'АО', 'ЗАО', 'ОАО', 'ПАО', 'ООО', 'Inc.', 'LLC', 'Ltd.', etc.
Return the full company name WITHOUT changing text.
If data not found return null.

Text: {text_section}

Counterparty name:""",
            
            'страна_контрагента': f"""Extract the counterparty's country. Counterparty is not Kazakhstan.
Look for country names near the counterparty name. DO NOT GUESS the countrty if not explicitly mentioned.
Return country code with dot separator and full country name, e.g. "RU.Российская Федерация".
If data not found return null.

Text: {text_section}

Country:""",
            
            'сумма_контракта': f"""Extract the total contract amount.
Look for "стоимость договора", "сумма", "составляет" followed by numbers.
Return ONLY the numeric value in float with 2 decimal places.
If data not found return null.

Text: {text_section}

Amount (number only):""",
            
            'валюта_контракта': f"""Extract the contract currency.
Look for "валюта", "рубли", "RUB", "USD", "EUR", etc.
Return standard currency code (RUB, USD, EUR, etc).
If data not found return null.

Text: {text_section}

Currency code:""",
            
            'валюта_платежа': f"""Extract the payment currency.
Return standard currency code (RUB, USD, EUR, etc). There can be multiple currencies separated by comma.
If data not found return null.

Text: {text_section}

Payment currency code:"""
        }
        
        return prompts.get(field, f"Extract {field} from: {text_section}")
    
    def predict_field_with_llm(self, field: str, text_section: str) -> Optional[Union[str, float]]:
        """Make a focused LLM call for a single field."""
        if not text_section or not text_section.strip():
            return None
        
        prompt = self.create_field_specific_prompt(field, text_section)
        
        try:
            response = chat(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a precise data extraction assistant. Return ONLY the requested value, no explanations.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                model=self.model_name,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'seed': 42
                }
            )
            
            result = response.message.content.strip()
            
            # Clean up the result
            if result.lower() in ['none', 'null', 'не найдено', 'not found', '']:
                return None
            
            # For amount field, try to convert to float with 2 decimal places
            if field == 'сумма_контракта':
                try:
                    # Remove non-numeric characters except dots and commas
                    cleaned = re.sub(r'[^\d.,]', '', result)
                    cleaned = cleaned.replace(',', '.')
                    amount = float(cleaned)
                    # If amount is 0 or negative, return null
                    if amount <= 0:
                        return None
                    # Format to 2 decimal places
                    return round(amount, 2)
                except:
                    return None
            
            # For date fields, validate format
            if 'дата' in field and result:
                # Try to ensure YYYY-MM-DD format
                date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', result)
                if date_match:
                    year, month, day = date_match.groups()
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            return result
            
        except Exception as e:
            print(f"Error extracting {field}: {e}")
            return None
    
    def predict_with_sections_from_json(self, json_data: List[Dict]) -> Dict[str, Optional[Union[str, float]]]:
        """Predict using section-based approach with separate LLM calls from JSON array."""
        # Extract relevant sections from JSON array
        sections = self.extract_relevant_sections_from_json(json_data)
        
        # Initialize result
        result = {
            "номер_контракта": None,
            "дата_контракта": None,
            "дата_окончания_контракта": None,
            "наименование_контрагента": None,
            "страна_контрагента": None,
            "сумма_контракта": None,
            "валюта_контракта": None,
            "валюта_платежа": None
        }
        
        # Process each field with its relevant section
        for field in result.keys():
            section = sections.get(field, '')
            if section:
                print(f"Extracting {field} from {len(section)} chars of filtered text...")
                value = self.predict_field_with_llm(field, section)
                if value is not None:
                    result[field] = value
            else:
                print(f"No matching text found for {field}")
        
        # Fallback: if валюта_платежа is null, copy from валюта_контракта
        if result["валюта_платежа"] is None and result["валюта_контракта"] is not None:
            print("валюта_платежа not found, using валюта_контракта as fallback")
            result["валюта_платежа"] = result["валюта_контракта"]
        
        return result
    
    def predict(self, text: str = None, json_data: List[Dict] = None) -> Optional[dict]:
        """Predict contract information from text or JSON array."""
        # If JSON array is provided, use section-based approach with LLM
        if json_data is not None and isinstance(json_data, list):
            print(f"Processing JSON array with section-based LLM approach...")
            
            # Check if model is available
            if self.check_model_available():
                llm_result = self.predict_with_sections_from_json(json_data)
                
                final_non_null = sum(1 for v in llm_result.values() if v is not None)
                print(f"Extraction complete: {final_non_null} fields found")
                
                return llm_result
            else:
                print(f"Error: Model {self.model_name} not available. Please ensure Ollama is running.")
                return None
        
        # If text is provided, process with LLM
        elif text:
            print(f"Processing text with LLM...")
            
            if self.check_model_available():
                # Create a simple sections dict with full text for each field
                sections = {field: text[:1000] for field in [
                    "номер_контракта", "дата_контракта", "дата_окончания_контракта",
                    "наименование_контрагента", "страна_контрагента",
                    "сумма_контракта", "валюта_контракта", "валюта_платежа"
                ]}
                
                result = {}
                for field, section in sections.items():
                    print(f"Extracting {field}...")
                    value = self.predict_field_with_llm(field, section)
                    result[field] = value
                
                # Fallback: if валюта_платежа is null, copy from валюта_контракта
                if result.get("валюта_платежа") is None and result.get("валюта_контракта") is not None:
                    print("валюта_платежа not found, using валюта_контракта as fallback")
                    result["валюта_платежа"] = result["валюта_контракта"]
                
                return result
            else:
                print(f"Error: Model {self.model_name} not available.")
                return None
        
        return None
    
    def process_file(self, file_path: str) -> Optional[dict]:
        """Process a file (text or JSON) and extract contract information."""
        try:
            path = Path(file_path)
            
            # Try to load as JSON first
            if path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        
                        # Check if it's already contract data (has expected fields)
                        if isinstance(json_data, dict) and 'номер_контракта' in json_data:
                            print(f"File {file_path} already contains contract data")
                            return json_data
                        
                        # Check if it's OCR output with full_text field
                        if isinstance(json_data, dict) and 'full_text' in json_data:
                            full_text = json_data['full_text']
                            print(f"Processing OCR full_text with {len(full_text)} characters...")
                            return self.predict(text=full_text)
                        
                        # If it's a list (array of text objects), use new approach
                        if isinstance(json_data, list):
                            print(f"Processing JSON array with {len(json_data)} items...")
                            return self.predict(json_data=json_data)
                        
                        # Otherwise, extract text from JSON structure
                        text = self.extract_text_from_json_array(json_data)
                        if text:
                            print(f"Extracted {len(text)} characters from JSON structure")
                            return self.predict(text=text)
                        else:
                            print(f"Warning: No text found in JSON file {file_path}")
                            return None
                            
                    except json.JSONDecodeError:
                        # If JSON parsing fails, treat as text file
                        f.seek(0)
                        text = f.read()
                        return self.predict(text=text)
            else:
                # Load as plain text
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if not text.strip():
                    print(f"Warning: File {file_path} is empty")
                    return None
                
                print(f"Processing {len(text)} characters of text...")
                return self.predict(text=text)
                
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def save_prediction(self, prediction: dict, output_path: str):
        """Save prediction to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(prediction, f, indent=2, ensure_ascii=False)
            print(f"Prediction saved to {output_path}")
        except Exception as e:
            print(f"Error saving prediction: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run Qwen3 predictions on OCR text using Ollama')
    parser.add_argument('input', help='Input file (text or JSON) or directory')
    parser.add_argument('--output-dir', default='predictions', help='Output directory for predictions')
    parser.add_argument('--model', default='gemma3:27b', help='Ollama model name')
    parser.add_argument('--full-text', action='store_true', help='Process full text from OCR output')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = OllamaQwen3Predictor(model_name=args.model)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        prediction = predictor.process_file(str(input_path))
        if prediction:
            # Generate output filename
            base_name = input_path.stem.replace('_ocr', '')
            output_file = output_dir / f"{base_name}.json"
            predictor.save_prediction(prediction, str(output_file))
        else:
            print("Failed to generate prediction")
            sys.exit(1)
            
    elif input_path.is_dir():
        # Process all files in directory
        # Check if directory name contains 'dots' (likely JSON processing)
        has_dots = 'dots' in input_path.name.lower()
        
        if has_dots:
            # For directories with 'dots', process JSON files
            files = list(input_path.glob("*.json"))
            print(f"Processing directory with 'dots' in name - looking for .json files")
        else:
            # Only process txt files without _ocr suffix and not inside _pages directories
            all_txt_files = list(input_path.glob("*.txt"))
            
            # Filter out files with _ocr suffix and files inside _pages directories
            files = []
            for file in all_txt_files:
                # Skip files with _ocr suffix
                if '_ocr' in file.stem:
                    continue
                # Skip files inside _pages directories
                if '_pages' in str(file.parent):
                    continue
                files.append(file)
        
        if not files:
            file_type = ".json" if has_dots else ".txt"
            print(f"No eligible {file_type} files found in {input_path}")
            if not has_dots:
                print(f"(Excluding files with '_ocr' suffix and files in '_pages' directories)")
            sys.exit(1)
        
        print(f"Found {len(files)} files to process")
        print(f"Predictions will be saved to: {output_dir.absolute()}/")
        
        success_count = 0
        for file in files:
            print(f"\nProcessing {file.name}...")
            prediction = predictor.process_file(str(file))
            
            if prediction:
                base_name = file.stem.replace('_ocr', '')
                output_file = output_dir / f"{base_name}.json"
                predictor.save_prediction(prediction, str(output_file))
                print(f"✓ Saved to: {output_file}")
                success_count += 1
            else:
                print(f"Failed to process {file.name}")
        
        print(f"\nProcessed {success_count}/{len(files)} files successfully")
        print(f"All predictions saved in: {output_dir.absolute()}/")
        
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()