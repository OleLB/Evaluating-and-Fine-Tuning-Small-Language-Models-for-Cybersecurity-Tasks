#!/usr/bin/env python3
"""
Convert training data from OpenAI format to Mistral format.

IMPORTANT: Mistral's apply_chat_template requires:
- Tool call IDs must be EXACTLY 9 alphanumeric characters
- Tool responses must reference these IDs with tool_call_id field

This is for training only - during inference, Mistral generates these IDs automatically.
"""

import json
import logging
import random
import string
from pathlib import Path
from typing import Dict, Any, List

# Configuration
INPUT_DIR = "LoRA/training_data_tool_calls_complete"
OUTPUT_DIR = "LoRA/training_data_tool_calls_complete_mistral"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MistralFormatConverter:
    """Convert OpenAI tool call format to Mistral format with proper IDs."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'files_converted': 0,
            'examples_converted': 0,
            'errors': 0
        }
    
    def generate_tool_call_id(self) -> str:
        """
        Generate a 9-character alphanumeric tool call ID for Mistral.
        
        Format: Exactly 9 characters, alphanumeric only
        Example: "AbCd12345"
        
        Returns:
            9-character string
        """
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(9))
    
    def convert_tool_call(self, tool_call: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        """
        Convert a single tool call from OpenAI to Mistral format.
        
        OpenAI format:
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "vector_database_retrieval",
                "arguments": "{\"query\": \"...\"}"
            }
        }
        
        Mistral format (for training):
        {
            "id": "AbCd12345",  # MUST be 9 alphanumeric chars
            "type": "function",
            "function": {
                "name": "vector_database_retrieval",
                "arguments": {"query": "..."}  # Dict, not string
            }
        }
        
        Returns:
            Tuple of (converted_tool_call, tool_call_id)
        """
        # Extract the function data
        function = tool_call.get('function', {})
        name = function.get('name')
        arguments = function.get('arguments')
        
        # Convert arguments from JSON string to dict if needed
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse arguments as JSON: {e}")
        
        # Generate a proper 9-character Mistral tool call ID
        mistral_id = self.generate_tool_call_id()
        
        # Return Mistral format WITH id and type (needed for apply_chat_template)
        return {
            "id": mistral_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments  # Dict, not string
            }
        }, mistral_id
    
    def convert_tool_message(self, message: Dict[str, Any], tool_call_id: str) -> Dict[str, Any]:
        """
        Convert a tool message to Mistral format.
        
        For training with apply_chat_template, Mistral DOES need tool_call_id.
        
        Args:
            message: Original tool message
            tool_call_id: The 9-character tool call ID to reference
        
        Returns:
            Converted tool message
        """
        return {
            "role": "tool",
            "name": message.get('name'),
            "content": message.get('content'),
            "tool_call_id": tool_call_id  # Required for Mistral's template
        }
    
    def convert_assistant_message(self, message: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """
        Convert an assistant message with tool calls.
        
        Returns:
            Tuple of (converted_message, list_of_tool_call_ids)
        """
        converted = {
            "role": "assistant"
        }
        
        tool_call_ids = []
        
        # If it has tool_calls, convert them
        if 'tool_calls' in message:
            converted_tool_calls = []
            
            for tc in message['tool_calls']:
                converted_tc, tc_id = self.convert_tool_call(tc)
                converted_tool_calls.append(converted_tc)
                tool_call_ids.append(tc_id)
            
            converted['tool_calls'] = converted_tool_calls
        
        # If it has content, include it
        if 'content' in message and message['content']:
            converted['content'] = message['content']
        
        return converted, tool_call_ids
    
    def convert_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a complete training example to Mistral format.
        
        This version maintains tool_call_id references for apply_chat_template.
        """
        if 'messages' not in example:
            raise ValueError("Example missing 'messages' field")
        
        converted_messages = []
        tool_call_id_map = {}  # Track tool call IDs for matching with tool responses
        
        for i, message in enumerate(example['messages']):
            role = message.get('role')
            
            if role == 'user':
                # User messages stay the same
                converted_messages.append({
                    "role": "user",
                    "content": message.get('content')
                })
            
            elif role == 'assistant':
                # Convert assistant messages (may have tool_calls)
                converted_msg, tool_call_ids = self.convert_assistant_message(message)
                converted_messages.append(converted_msg)
                
                # Store tool call IDs for the next tool message
                if tool_call_ids:
                    tool_call_id_map[i] = tool_call_ids
            
            elif role == 'tool':
                # Find the corresponding tool call ID
                # Look backwards for the most recent assistant message with tool calls
                tool_call_id = None
                for prev_idx in range(i - 1, -1, -1):
                    if prev_idx in tool_call_id_map:
                        # Get the first unused tool call ID
                        if tool_call_id_map[prev_idx]:
                            tool_call_id = tool_call_id_map[prev_idx].pop(0)
                            break
                
                if not tool_call_id:
                    # Generate a new ID if we couldn't find one
                    logger.warning(f"Could not find matching tool_call_id, generating new one")
                    tool_call_id = self.generate_tool_call_id()
                
                # Convert tool response message
                converted_messages.append(
                    self.convert_tool_message(message, tool_call_id)
                )
            
            else:
                logger.warning(f"Unknown role: {role}")
                converted_messages.append(message)
        
        return {
            "messages": converted_messages
        }
    
    def convert_file(self, input_path: Path) -> bool:
        """Convert a single JSONL file to Mistral format."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting: {input_path.name}")
        logger.info(f"{'='*60}")
        
        output_path = self.output_dir / input_path.name
        converted_examples = []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        example = json.loads(line)
                        converted = self.convert_example(example)
                        converted_examples.append(converted)
                        
                        logger.debug(f"  Line {line_num}: Converted successfully")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"  Line {line_num}: Invalid JSON - {e}")
                        self.stats['errors'] += 1
                    except Exception as e:
                        logger.error(f"  Line {line_num}: Conversion failed - {e}")
                        self.stats['errors'] += 1
            
            # Write converted examples
            if converted_examples:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for example in converted_examples:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
                
                logger.info(f"✓ Converted {len(converted_examples)} examples")
                logger.info(f"  Saved to: {output_path.name}")
                
                # Show a sample conversion
                if converted_examples:
                    logger.info("\nSample converted example:")
                    logger.info(json.dumps(converted_examples[0], indent=2)[:500] + "...")
                
                self.stats['files_converted'] += 1
                self.stats['examples_converted'] += len(converted_examples)
                return True
            else:
                logger.warning(f"✗ No examples converted from {input_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Failed to process file {input_path.name}: {e}")
            return False
    
    def convert_all(self):
        """Convert all JSONL files in the input directory."""
        jsonl_files = list(self.input_dir.glob("*.jsonl"))
        json_files = list(self.input_dir.glob("*.json"))
        all_files = jsonl_files + json_files
        
        if not all_files:
            logger.error(f"No JSON/JSONL files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(all_files)} file(s) to convert")
        logger.info(f"Input directory:  {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}\n")
        
        for input_path in all_files:
            self.convert_file(input_path)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("Conversion Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Files converted:    {self.stats['files_converted']}")
        logger.info(f"Examples converted: {self.stats['examples_converted']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info(f"Output directory:   {self.output_dir}")
        logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    converter = MistralFormatConverter(INPUT_DIR, OUTPUT_DIR)
    converter.convert_all()


if __name__ == "__main__":
    main()