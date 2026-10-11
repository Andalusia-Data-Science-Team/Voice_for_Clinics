import os
import logging
from datetime import datetime

def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger with time and process id
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(f'logs/app-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatters and add them to handlers
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - [%(process)d] - %(name)s - %(message)s')
    
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# model/utils.py
import json
import re

def safe_parse_json(raw: str) -> dict | list:
    text = raw.strip()

    # Handle case where the whole thing is a JSON string containing fenced JSON
    # e.g. "{\"questions\": \"```json\\n{...}\\n```\"}"
    if text.startswith('"') and text.endswith('"'):
        try:
            inner = json.loads(text)  # unwrap outer string
            text = inner.strip()
        except json.JSONDecodeError:
            pass

    # Strip markdown code fences (handles ```json, ```JSON, ``` variants)
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try closing unterminated structures (truncation recovery)
    repaired = _repair_truncated_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unrecoverable JSON from LLM: {e}\nRaw: {raw[:200]}")


def _repair_truncated_json(text: str) -> str:
    """Close any unclosed brackets/braces/strings caused by truncation."""
    # Remove trailing incomplete key-value or dangling comma
    text = re.sub(r',\s*$', '', text.rstrip())
    text = re.sub(r',\s*[}\]]', lambda m: m.group(0)[-1], text)  # trailing comma before close

    # Count open structures
    stack = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in '{[':
            stack.append('}' if char == '{' else ']')
        elif char in '}]' and stack:
            stack.pop()

    # If we're mid-string, close it
    if in_string:
        text += '"'

    # Close open structures in reverse order
    text += ''.join(reversed(stack))

    return text
