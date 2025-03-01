#!/usr/bin/env python
"""
Test script for the streaming API.
This script sends a test request to the API with streaming enabled and prints the response as it arrives.
"""

import requests
import json
import time
import argparse
from urllib.parse import urljoin
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the streaming API")
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
        default=0.5,
        help="Temperature for generation",
    )
    return parser.parse_args()

def test_streaming_api(host, port, prompt, temperature):
    """Test the streaming API by sending a request and printing the response as it arrives."""
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
        "stream": True,
    }
    
    # Send request
    print(f"Sending streaming request to {url}...")
    print(f"Prompt: {prompt}")
    print(f"Temperature: {temperature}")
    print("\nResponse:")
    
    start_time = time.time()
    try:
        with requests.post(url, json=data, stream=True, timeout=120) as response:
            response.raise_for_status()
            
            full_text = ""
            for line in response.iter_lines():
                if line:
                    # Parse the JSON response
                    chunk = json.loads(line)
                    
                    # Print the token
                    if chunk["token"]:
                        sys.stdout.write(chunk["token"])
                        sys.stdout.flush()
                        full_text += chunk["token"]
                    
                    # If this is the final chunk, print the metadata
                    if chunk["finished"] and chunk["metadata"]:
                        print("\n\nMetadata:")
                        for key, value in chunk["metadata"].items():
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
    
    # Test streaming endpoint
    test_streaming_api(args.host, args.port, args.prompt, args.temperature) 