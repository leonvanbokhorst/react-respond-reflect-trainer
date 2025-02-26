"""
VLLM API Client for React-Respond-Reflect Model

This script demonstrates how to interact with the VLLM API server running your RRR model.
It provides a simple command-line interface for chatting with the model.

Requirements:
pip install requests rich
"""

import argparse
import json
import re
import sys
import time
from typing import List, Dict, Optional

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

# Custom theme for output formatting
custom_theme = Theme({
    "user": "bold cyan",
    "assistant": "bold green",
    "react": "italic yellow",
    "respond": "bold white",
    "reflect": "italic magenta",
    "error": "bold red",
    "timing": "dim blue",
})

console = Console(theme=custom_theme)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="VLLM API Client for RRR Model")
    parser.add_argument(
        "--api_url", 
        type=str, 
        default="http://localhost:8000/v1/chat/completions",
        help="URL for the VLLM API server"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--system_prompt", 
        type=str, 
        default="You are an empathetic AI assistant. Always structure your response with: "
                "<react>*your internal thought process*</react> first, followed by "
                "<respond>your direct response to the user</respond>, and finally "
                "<reflect>your reflection on this interaction</reflect>.",
        help="System prompt to use"
    )
    return parser.parse_args()


def extract_rrr_sections(text: str) -> Dict[str, str]:
    """
    Extract React, Respond, and Reflect sections from generated text.
    
    Args:
        text: The generated text
        
    Returns:
        dict: Dictionary with 'react', 'respond', and 'reflect' keys
    """
    sections = {}
    
    # Extract each section with regex
    react_match = re.search(r'<react>\s*(.*?)\s*</react>', text, re.DOTALL)
    respond_match = re.search(r'<respond>\s*(.*?)\s*</respond>', text, re.DOTALL)
    reflect_match = re.search(r'<reflect>\s*(.*?)\s*</reflect>', text, re.DOTALL)
    
    # Add matches to sections dict
    if react_match:
        sections['react'] = react_match.group(1).strip()
    else:
        sections['react'] = "No react section found"
        
    if respond_match:
        sections['respond'] = respond_match.group(1).strip()
    else:
        sections['respond'] = "No respond section found"
        
    if reflect_match:
        sections['reflect'] = reflect_match.group(1).strip()
    else:
        sections['reflect'] = "No reflect section found"
    
    return sections


def generate_completion(api_url: str, messages: List[Dict], args) -> Dict:
    """
    Generate a completion from the API.
    
    Args:
        api_url: URL of the API
        messages: List of messages
        args: Command-line arguments
        
    Returns:
        dict: API response
    """
    payload = {
        "model": "rrr_model",  # Model ID (can be anything for VLLM server)
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[error]Error: {e}[/error]")
        return None


def display_response(response: Dict, show_react=True, show_reflect=True):
    """
    Display the model's response with formatting.
    
    Args:
        response: The API response
        show_react: Whether to show the react section
        show_reflect: Whether to show the reflect section
    """
    if not response or 'choices' not in response:
        console.print("[error]No valid response received[/error]")
        return
    
    # Get the assistant's message
    assistant_message = response['choices'][0]['message']['content']
    
    # Extract RRR sections
    sections = extract_rrr_sections(assistant_message)
    
    # Display timing information
    usage = response.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    
    console.print(f"[timing]Tokens: {prompt_tokens} prompt + {completion_tokens} completion[/timing]")
    
    # Display the formatted response
    if show_react:
        console.print(Panel(
            Markdown(sections['react']),
            title="[react]React (Internal Thoughts)[/react]",
            border_style="yellow"
        ))
    
    console.print(Panel(
        Markdown(sections['respond']),
        title="[respond]Response[/respond]",
        border_style="green"
    ))
    
    if show_reflect:
        console.print(Panel(
            Markdown(sections['reflect']),
            title="[reflect]Reflect[/reflect]",
            border_style="magenta"
        ))


def interactive_chat(args):
    """Run an interactive chat session with the API"""
    console.print("[bold]React-Respond-Reflect Model Chat[/bold]")
    console.print("Type your messages (Ctrl+D or type 'exit' to quit)")
    console.print("Use /togglereact to hide/show React sections")
    console.print("Use /togglereflect to hide/show Reflect sections")
    
    # Initialize messages with system prompt
    messages = [{"role": "system", "content": args.system_prompt}]
    
    # Visualization controls
    show_react = True
    show_reflect = True
    
    while True:
        try:
            # Get user input
            console.print("\n[user]You:[/user] ", end="")
            user_input = input()
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                break
                
            # Check for toggle commands
            if user_input.lower() == "/togglereact":
                show_react = not show_react
                console.print(f"React sections {'shown' if show_react else 'hidden'}")
                continue
                
            if user_input.lower() == "/togglereflect":
                show_reflect = not show_reflect
                console.print(f"Reflect sections {'shown' if show_reflect else 'hidden'}")
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get response
            console.print("\n[assistant]Assistant:[/assistant]")
            start_time = time.time()
            response = generate_completion(args.api_url, messages, args)
            generation_time = time.time() - start_time
            
            # Display timing
            console.print(f"[timing]Generation time: {generation_time:.2f}s[/timing]")
            
            # Display response
            display_response(response, show_react, show_reflect)
            
            # Add assistant message to history
            if response and 'choices' in response:
                messages.append(response['choices'][0]['message'])
                
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("\nExiting...")
            break
    
    console.print("Chat session ended.")


def main():
    """Main entry point"""
    args = parse_args()
    interactive_chat(args)


if __name__ == "__main__":
    main()
