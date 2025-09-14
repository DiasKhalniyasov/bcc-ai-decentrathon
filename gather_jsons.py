#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def gather_jsons(input_dir, output_file="combined_jsons.json"):
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return False
    
    if not input_path.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        return False
    
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}'")
        return False
    
    print(f"Found {len(json_files)} JSON files to process")
    
    combined_data = []
    
    for json_file in sorted(json_files):
        print(f"Processing: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)
        except json.JSONDecodeError as e:
            print(f"  Warning: Failed to parse {json_file.name}: {e}")
        except Exception as e:
            print(f"  Warning: Error reading {json_file.name}: {e}")
    
    output_path = Path(output_file)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully combined {len(combined_data)} JSON files into '{output_file}'")
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python gather_jsons.py <input_directory> [output_file]")
        print("  input_directory: Path to directory containing JSON files")
        print("  output_file: Optional output filename (default: combined_jsons.json)")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "combined_jsons.json"
    
    success = gather_jsons(input_dir, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()