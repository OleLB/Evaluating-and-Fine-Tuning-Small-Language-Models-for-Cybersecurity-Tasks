#!/usr/bin/env python3
"""
Convert JSON files to JSONL format.

This script:
1. Finds all .json files in a specified folder
2. Converts JSON arrays to JSONL (one object per line)
3. Saves as .jsonl files in the same or different folder
4. Handles both single objects and arrays
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_json_to_jsonl(input_file: Path, output_file: Path) -> bool:
    """
    Convert a JSON file to JSONL format.
    
    Args:
        input_file: Path to input .json file
        output_file: Path to output .jsonl file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # JSON array - each item becomes one line
            examples = data
        elif isinstance(data, dict):
            # Single JSON object - becomes one line
            examples = [data]
        else:
            print(f"  ✗ Unsupported JSON structure in {input_file.name}")
            return False
        
        # Write as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"  ✓ {input_file.name} → {output_file.name} ({len(examples)} examples)")
        return True
        
    except json.JSONDecodeError as e:
        print(f"  ✗ Invalid JSON in {input_file.name}: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error processing {input_file.name}: {e}")
        return False


def convert_folder(
    input_folder: str, 
    output_folder: str = None,
    overwrite: bool = False,
    remove_original: bool = False
):
    """
    Convert all .json files in a folder to .jsonl format.
    
    Args:
        input_folder: Folder containing .json files
        output_folder: Folder for .jsonl files (None = same as input)
        overwrite: Whether to overwrite existing .jsonl files
        remove_original: Whether to delete original .json files after conversion
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Folder '{input_folder}' does not exist")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory")
        return
    
    # Set output folder
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Find all .json files
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No .json files found in {input_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"JSON to JSONL Conversion")
    print(f"{'='*60}")
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder or input_folder}")
    print(f"Found {len(json_files)} .json file(s)\n")
    
    # Track statistics
    stats = {
        'converted': 0,
        'skipped': 0,
        'failed': 0,
        'deleted': 0
    }
    
    # Convert each file
    for json_file in json_files:
        # Create output filename
        output_file = output_path / f"{json_file.stem}.jsonl"
        
        # Check if output already exists
        if output_file.exists() and not overwrite:
            print(f"  ⊘ {json_file.name} - output exists (use --overwrite to replace)")
            stats['skipped'] += 1
            continue
        
        # Convert the file
        success = convert_json_to_jsonl(json_file, output_file)
        
        if success:
            stats['converted'] += 1
            
            # Remove original if requested
            if remove_original:
                try:
                    json_file.unlink()
                    print(f"    🗑 Deleted {json_file.name}")
                    stats['deleted'] += 1
                except Exception as e:
                    print(f"    ⚠ Could not delete {json_file.name}: {e}")
        else:
            stats['failed'] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"Successfully converted: {stats['converted']}")
    print(f"Skipped (exists):       {stats['skipped']}")
    print(f"Failed:                 {stats['failed']}")
    if remove_original:
        print(f"Original files deleted: {stats['deleted']}")
    print(f"{'='*60}\n")


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert JSON files to JSONL format (one object per line)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all .json files in current folder to .jsonl
  python json_to_jsonl.py .

  # Convert with specific input/output folders
  python json_to_jsonl.py data/json_files -o data/jsonl_files

  # Overwrite existing .jsonl files
  python json_to_jsonl.py data -w

  # Convert and delete original .json files
  python json_to_jsonl.py data --remove-original

  # Preview what would be converted (dry run)
  python json_to_jsonl.py data --dry-run
        """
    )
    
    parser.add_argument(
        'input_folder',
        help='Folder containing .json files to convert'
    )
    
    parser.add_argument(
        '-o', '--output-folder',
        help='Output folder for .jsonl files (default: same as input)',
        default=None
    )
    
    parser.add_argument(
        '-w', '--overwrite',
        action='store_true',
        help='Overwrite existing .jsonl files'
    )
    
    parser.add_argument(
        '--remove-original',
        action='store_true',
        help='Delete original .json files after conversion'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be converted without actually converting'
    )
    
    args = parser.parse_args()
    
    # Dry run mode
    if args.dry_run:
        input_path = Path(args.input_folder)
        json_files = list(input_path.glob("*.json"))
        
        print(f"\nDRY RUN - No files will be converted\n")
        print(f"Found {len(json_files)} .json file(s) in {args.input_folder}:")
        
        for json_file in json_files:
            output_name = f"{json_file.stem}.jsonl"
            print(f"  {json_file.name} → {output_name}")
        
        print(f"\nRun without --dry-run to perform conversion")
        return
    
    # Perform conversion
    convert_folder(
        args.input_folder,
        args.output_folder,
        args.overwrite,
        args.remove_original
    )


if __name__ == "__main__":
    main()