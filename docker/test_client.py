#!/usr/bin/env python3
"""
Test client for the React-Respond-Reflect vLLM API.

This script provides a simple command-line interface to test
the deployed RRR model API. It supports both the OpenAI-compatible
endpoint and the custom RRR endpoint.

Example:
    $ python test_client.py --endpoint http://localhost:8000
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

import requests


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test RRR model API")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["openai", "rrr"],
        default="rrr",
        help="API type to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    return parser.parse_args()


def chat_loop(
    endpoint: str,
    api_type: str,
    temperature: float,
    max_tokens: int,
) -> None:
    """
    Interactive chat loop with the RRR model.

    Args:
        endpoint: API endpoint URL
        api_type: API type to use (openai or rrr)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    print("🤖 React-Respond-Reflect Chat")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 50)

    # Initialize conversation history
    messages = []

    while True:
        # Get user input
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Prepare request based on API type
        if api_type == "openai":
            url = f"{endpoint}/v1/chat/completions"
            payload = {
                "model": "rrr-model",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        else:  # rrr
            url = f"{endpoint}/rrr/chat"
            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        # Send request
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract and display response
            if api_type == "openai":
                assistant_message = result["choices"][0]["message"]["content"]
                print(f"\n🤖 Assistant:\n{assistant_message}")
                messages.append({"role": "assistant", "content": assistant_message})
            else:  # rrr
                print("\n🤖 Assistant:")
                if "components" in result and all(result["components"].values()):
                    react = result["components"]["react"]
                    respond = result["components"]["respond"]
                    reflect = result["components"]["reflect"]

                    print(f"  React: *{react}*")
                    print(f"  Respond: {respond}")
                    print(f"  Reflect: {reflect}")
                else:
                    print(result["content"])

                messages.append({"role": "assistant", "content": result["content"]})

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if hasattr(e, "response") and hasattr(e.response, "text"):
                print(f"Response: {e.response.text}")


def main():
    """Main function."""
    args = parse_args()

    # Check if the server is running
    try:
        health_response = requests.get(f"{args.endpoint}/health")
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"✅ Server is running with model: {health_data.get('model', 'unknown')}")
    except Exception as e:
        print(f"❌ Error connecting to server: {str(e)}")
        print(f"Make sure the server is running at {args.endpoint}")
        sys.exit(1)

    # Start chat loop
    chat_loop(
        endpoint=args.endpoint,
        api_type=args.api,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
