#!/usr/bin/env python
"""
Test script for the model API.
This script sends a test request to the API and prints the response.
"""

import requests
import json
import time
import argparse
from urllib.parse import urljoin

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the model API")
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
    return parser.parse_args()

def test_api(host, port, prompt):
    """Test the API by sending a request and printing the response."""
    api_url = f"http://{host}:{port}"
    endpoint = "/generate"
    url = urljoin(api_url, endpoint)
    
    # Prepare request data
    data = {
        "prompt": prompt,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }
    
    # Send request
    print(f"Sending request to {url}...")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Print response
        print("\nResponse:")
        print(f"Generated text: {result['generated_text']}")
        print("\nMetadata:")
        for key, value in result["metadata"].items():
            print(f"  {key}: {value}")
        
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
    test_api(args.host, args.port, args.prompt) 