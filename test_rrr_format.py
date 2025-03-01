#!/usr/bin/env python
"""
Test script to check if the model is generating responses in the correct RRR format.
This script sends a test request to the API and validates the response format.
"""

import requests
import json
import re
import time
import argparse
from urllib.parse import urljoin
from typing import Dict, Any

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the model API for RRR format")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="API host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7000,
        help="API port",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="I'm feeling stressed about my upcoming presentation. Can you help?",
        help="Prompt to send to the API",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    return parser.parse_args()

def validate_rrr_format(text: str) -> Dict[str, Any]:
    """
    Validate that responses follow React-Respond-Reflect format.
    
    Args:
        text: The text to validate
            
    Returns:
        Dict with validation results
    """
    # Extract assistant response
    assistant_text = text
    
    # Check for all three tags
    react_match = re.search(r'<react>\s*(.*?)\s*</react>', assistant_text, re.DOTALL)
    respond_match = re.search(r'<respond>\s*(.*?)\s*</respond>', assistant_text, re.DOTALL)
    reflect_match = re.search(r'<reflect>\s*(.*?)\s*</reflect>', assistant_text, re.DOTALL)
    
    has_all_tags = all([react_match, respond_match, reflect_match])
    
    # Check order if all tags are present
    correct_order = False
    if has_all_tags:
        react_pos = assistant_text.find('<react>')
        respond_pos = assistant_text.find('<respond>')
        reflect_pos = assistant_text.find('<reflect>')
        correct_order = (react_pos < respond_pos < reflect_pos)
    
    # Check content in each section
    tag_content = {
        "react": bool(react_match and react_match.group(1).strip()),
        "respond": bool(respond_match and respond_match.group(1).strip()),
        "reflect": bool(reflect_match and reflect_match.group(1).strip()),
    }
    
    return {
        "valid": has_all_tags and correct_order,
        "has_all_tags": has_all_tags,
        "correct_order": correct_order,
        "tag_content": tag_content
    }

def test_api(host, port, prompt, temperature):
    """Test the API by sending a request and validating the response format."""
    api_url = f"http://{host}:{port}"
    endpoint = "/generate"
    url = urljoin(api_url, endpoint)
    
    # Prepare request data
    data = {
        "prompt": prompt,
        "max_new_tokens": 512,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }
    
    # Send request
    print(f"Sending request to {url}...")
    print(f"Prompt: {prompt}")
    print(f"Temperature: {temperature}")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=120)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Print response
        print("\nResponse:")
        print(f"Generated text: {result['generated_text']}")
        print("\nMetadata:")
        for key, value in result["metadata"].items():
            print(f"  {key}: {value}")
        
        # Validate RRR format
        format_validation = validate_rrr_format(result["generated_text"])
        print("\nRRR Format Validation:")
        for key, value in format_validation.items():
            print(f"  {key}: {value}")
        
        if format_validation["valid"]:
            print("\n✅ Response follows the RRR format!")
        else:
            print("\n❌ Response does NOT follow the RRR format!")
            
            if not format_validation["has_all_tags"]:
                print("   Missing tags:")
                if not format_validation["tag_content"]["react"]:
                    print("   - <react> tag missing or empty")
                if not format_validation["tag_content"]["respond"]:
                    print("   - <respond> tag missing or empty")
                if not format_validation["tag_content"]["reflect"]:
                    print("   - <reflect> tag missing or empty")
            
            if not format_validation["correct_order"]:
                print("   Tags are not in the correct order (should be: <react>, <respond>, <reflect>)")
        
        print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
        
    except requests.RequestException as e:
        print(f"Error: {str(e)}")
        return False
    
    return True

def test_health(host, port):
    """Test the health endpoint."""
    api_url = f"http://{host}:{port}"
    endpoint = "/health"
    url = urljoin(api_url, endpoint)
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        print(f"Health check: {result}")
        return result.get("status") == "ok"
        
    except requests.RequestException as e:
        print(f"Health check failed: {str(e)}")
        return False

if __name__ == "__main__":
    args = parse_args()
    
    # Test health endpoint
    if test_health(args.host, args.port):
        print("Health check passed!")
    else:
        print("Health check failed!")
        exit(1)
    
    # Test generate endpoint
    test_api(args.host, args.port, args.prompt, args.temperature) 