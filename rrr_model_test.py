"""
Test script for evaluating the React-Respond-Reflect model.

This script:
1. Loads the best saved model from the rrr_model directory
2. Tests it with a set of prompts
3. Validates the React-Respond-Reflect format
4. Provides basic metrics on response quality and format compliance
5. Tests multi-turn conversations
"""

import torch
import re
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import TextStreamer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class RRRModelTester:
    """Tester for React-Respond-Reflect models."""
    
    def __init__(
        self,
        model_path: str = "rrr_model",
        max_new_tokens: int = 2048,
        test_prompts: List[str] = None,
        test_conversations: List[Dict] = None,
    ):
        """
        Initialize the model tester.
        
        Args:
            model_path: Path to the model directory
            max_new_tokens: Maximum tokens to generate
            test_prompts: List of prompts to test (if None, default prompts will be used)
            test_conversations: List of multi-turn conversations to test
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Default test prompts if none provided
        self.test_prompts = test_prompts or [
            "I'm feeling really stressed about my upcoming presentation. Can you help?",
            "Tell me about a time you learned something new. How did it feel?",
            "I'm not sure if I'm making the right career choice. Any advice?",
            "I've been feeling down lately and I'm not sure why.",
            "How do you handle criticism?",
            "I'm thinking about starting a new hobby. Any suggestions?",
            "What's your approach to solving complex problems?",
            "I'm having trouble sleeping lately. Do you have any tips?",
            "I'm nervous about meeting new people at an event tomorrow.",
            "How do you stay motivated when working on long-term goals?"
        ]
        
        # Default test conversations if none provided
        self.test_conversations = test_conversations or [
            {
                "name": "Career Advice Conversation",
                "turns": [
                    "I'm thinking about changing careers. I've been in marketing for 5 years but I'm interested in data science.",
                    "I'm worried about starting over. Do you think it's too late to switch?",
                    "That's helpful. What skills should I focus on learning first?"
                ]
            },
            {
                "name": "Anxiety Management Conversation",
                "turns": [
                    "I've been feeling really anxious lately, especially in social situations.",
                    "I think it started after a bad experience at a work presentation last month.",
                    "What are some techniques I could try to manage this anxiety?"
                ]
            }
        ]
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"🔄 Loading model from {self.model_path}...")
        
        # Load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=2048,
            #load_in_4bit=True,
        )
        
        # Prepare model for inference
        FastLanguageModel.for_inference(self.model)
        print("✅ Model loaded successfully!")
    
    def validate_rrr_format(self, text: str) -> Dict[str, Union[bool, Dict[str, bool]]]:
        """
        Validate that assistant responses follow React-Respond-Reflect format.
        
        Args:
            text: The text to validate
            
        Returns:
            Dict with validation results
        """
        # Split into turns
        turns = text.split("<|im_start|>assistant\n")
        
        if len(turns) <= 1:
            return {
                "valid": False,
                "reason": "No assistant response found",
                "components": {
                    "react": False,
                    "respond": False,
                    "reflect": False,
                    "correct_order": False
                }
            }
        
        # Check the last turn (most recent assistant response)
        turn = turns[-1]
        
        # Check for all three tags
        react_match = re.search(r'<react>\s*(.*?)\s*</react>', turn, re.DOTALL)
        respond_match = re.search(r'<respond>\s*(.*?)\s*</respond>', turn, re.DOTALL)
        reflect_match = re.search(r'<reflect>\s*(.*?)\s*</reflect>', turn, re.DOTALL)
        
        components = {
            "react": bool(react_match),
            "respond": bool(respond_match),
            "reflect": bool(reflect_match),
            "correct_order": False
        }
        
        # Check if all components exist
        if not all([react_match, respond_match, reflect_match]):
            return {
                "valid": False,
                "reason": "Missing one or more RRR components",
                "components": components
            }
            
        # Check order
        react_pos = turn.find('<react>')
        respond_pos = turn.find('<respond>')
        reflect_pos = turn.find('<reflect>')
        
        correct_order = (react_pos < respond_pos < reflect_pos)
        components["correct_order"] = correct_order
        
        if not correct_order:
            return {
                "valid": False,
                "reason": "RRR components in incorrect order",
                "components": components
            }
        
        # Extract content from each component
        react_content = react_match.group(1).strip()
        respond_content = respond_match.group(1).strip()
        reflect_content = reflect_match.group(1).strip()
        
        # Check for minimum content length (at least 10 characters)
        min_length = 10
        if len(react_content) < min_length or len(respond_content) < min_length or len(reflect_content) < min_length:
            return {
                "valid": False,
                "reason": "One or more RRR components has insufficient content",
                "components": components
            }
        
        return {
            "valid": True,
            "components": components,
            "content": {
                "react": react_content,
                "respond": respond_content,
                "reflect": reflect_content
            }
        }
    
    def generate_response(self, prompt: str, conversation_history: List[Dict] = None) -> Tuple[str, float, Dict]:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: The prompt to generate a response for
            conversation_history: Optional conversation history for multi-turn conversations
            
        Returns:
            Tuple of (response text, generation time, format validation results)
        """
        # Prepare input
        if conversation_history:
            messages = conversation_history + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
            
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        start_time = time.time()
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=self.max_new_tokens,
            use_cache=True
        )
        generation_time = time.time() - start_time
        
        # Decode response
        output_text = self.tokenizer.decode(outputs[0])
        
        # Validate format
        format_validation = self.validate_rrr_format(output_text)
        
        return output_text, generation_time, format_validation
    
    def run_tests(self) -> Dict:
        """
        Run tests on all prompts.
        
        Returns:
            Dict with test results
        """
        if self.model is None:
            self.load_model()
        
        results = {
            "model_path": self.model_path,
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_results": [],
            "conversation_results": [],
            "summary": {
                "total_prompts": len(self.test_prompts),
                "total_conversations": len(self.test_conversations),
                "format_valid_count": 0,
                "avg_generation_time": 0,
                "component_success": {
                    "react": 0,
                    "respond": 0,
                    "reflect": 0,
                    "correct_order": 0
                }
            }
        }
        
        total_time = 0
        total_responses = 0
        
        # Test single prompts
        print(f"🧪 Testing model with {len(self.test_prompts)} prompts...")
        for i, prompt in enumerate(tqdm(self.test_prompts)):
            print(f"\n\nPrompt {i+1}/{len(self.test_prompts)}: {prompt}")
            
            # Generate and validate response
            response, gen_time, validation = self.generate_response(prompt)
            
            # Extract assistant response only
            assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
            print(f"\nResponse:\n{assistant_response}")
            
            # Update results
            prompt_result = {
                "prompt": prompt,
                "response": assistant_response,
                "generation_time": gen_time,
                "format_validation": validation
            }
            
            results["prompt_results"].append(prompt_result)
            total_time += gen_time
            total_responses += 1
            
            # Update summary stats
            if validation["valid"]:
                results["summary"]["format_valid_count"] += 1
            
            for component, success in validation["components"].items():
                if success:
                    results["summary"]["component_success"][component] += 1
        
        # Test multi-turn conversations
        print(f"\n🔄 Testing model with {len(self.test_conversations)} multi-turn conversations...")
        for conv_idx, conversation in enumerate(self.test_conversations):
            print(f"\n\nConversation {conv_idx+1}: {conversation['name']}")
            
            conv_result = {
                "name": conversation["name"],
                "turns": []
            }
            
            # Initialize conversation history
            history = []
            
            # Process each turn
            for turn_idx, user_message in enumerate(conversation["turns"]):
                print(f"\nTurn {turn_idx+1}: {user_message}")
                
                # Generate response
                response, gen_time, validation = self.generate_response(user_message, history)
                
                # Extract assistant response
                assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
                print(f"Response:\n{assistant_response}")
                
                # Update conversation history
                history.append({"role": "user", "content": user_message})
                history.append({"role": "assistant", "content": assistant_response})
                
                # Update turn results
                turn_result = {
                    "user_message": user_message,
                    "assistant_response": assistant_response,
                    "generation_time": gen_time,
                    "format_validation": validation
                }
                
                conv_result["turns"].append(turn_result)
                
                # Update summary stats
                total_time += gen_time
                total_responses += 1
                
                if validation["valid"]:
                    results["summary"]["format_valid_count"] += 1
                
                for component, success in validation["components"].items():
                    if success:
                        results["summary"]["component_success"][component] += 1
            
            results["conversation_results"].append(conv_result)
        
        # Calculate averages
        results["summary"]["avg_generation_time"] = total_time / total_responses
        
        # Calculate percentages
        total = total_responses
        results["summary"]["format_valid_percent"] = (results["summary"]["format_valid_count"] / total) * 100
        
        for component in results["summary"]["component_success"]:
            count = results["summary"]["component_success"][component]
            results["summary"]["component_success"][component] = {
                "count": count,
                "percent": (count / total) * 100
            }
        
        return results
    
    def generate_report(self, results: Dict):
        """
        Generate a report from test results.
        
        Args:
            results: Test results dictionary
        """
        print("\n📊 Test Results Summary:")
        print(f"Model: {results['model_path']}")
        print(f"Test Time: {results['test_time']}")
        print(f"Total Prompts: {results['summary']['total_prompts']}")
        print(f"Total Conversations: {results['summary']['total_conversations']}")
        print(f"Total Responses: {results['summary']['total_prompts'] + sum(len(conv['turns']) for conv in results['conversation_results'])}")
        print(f"Format Valid: {results['summary']['format_valid_count']} ({results['summary']['format_valid_percent']:.1f}%)")
        print(f"Average Generation Time: {results['summary']['avg_generation_time']:.2f} seconds")
        
        print("\nComponent Success Rates:")
        for component, stats in results["summary"]["component_success"].items():
            print(f"  - {component}: {stats['count']} ({stats['percent']:.1f}%)")
        
        # Save results to file
        results_file = self.results_dir / f"test_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to {results_file}")
        
        # Generate visualizations
        self._generate_visualizations(results)
    
    def _generate_visualizations(self, results: Dict):
        """
        Generate visualizations from test results.
        
        Args:
            results: Test results dictionary
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Format compliance pie chart
        labels = ['Valid Format', 'Invalid Format']
        sizes = [
            results['summary']['format_valid_count'],
            results['summary']['total_prompts'] + sum(len(conv['turns']) for conv in results['conversation_results']) - results['summary']['format_valid_count']
        ]
        colors = ['#4CAF50', '#F44336']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Format Compliance')
        
        # Component success rates bar chart
        components = list(results['summary']['component_success'].keys())
        success_rates = [stats['percent'] for stats in results['summary']['component_success'].values()]
        
        sns.barplot(x=components, y=success_rates, palette='viridis', ax=ax2)
        ax2.set_ylim(0, 100)
        ax2.set_title('Component Success Rates (%)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_xlabel('Component')
        
        for i, rate in enumerate(success_rates):
            ax2.text(i, rate + 2, f"{rate:.1f}%", ha='center')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.results_dir / f"test_viz_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_file)
        print(f"Visualizations saved to {viz_file}")
        
        # Generation time distribution
        plt.figure(figsize=(10, 6))
        
        # Collect all generation times
        gen_times = [result['generation_time'] for result in results['prompt_results']]
        for conv in results['conversation_results']:
            for turn in conv['turns']:
                gen_times.append(turn['generation_time'])
                
        sns.histplot(gen_times, kde=True)
        plt.title('Generation Time Distribution')
        plt.xlabel('Generation Time (seconds)')
        plt.ylabel('Frequency')
        
        # Save visualization
        time_viz_file = self.results_dir / f"time_viz_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(time_viz_file)
        print(f"Time distribution visualization saved to {time_viz_file}")
        
        # Multi-turn conversation performance
        if results['conversation_results']:
            plt.figure(figsize=(12, 6))
            
            # Prepare data
            conv_names = []
            turn_counts = []
            valid_counts = []
            
            for conv in results['conversation_results']:
                conv_names.append(conv['name'])
                turn_counts.append(len(conv['turns']))
                valid_counts.append(sum(1 for turn in conv['turns'] if turn['format_validation']['valid']))
            
            # Create grouped bar chart
            x = range(len(conv_names))
            width = 0.35
            
            plt.bar(x, turn_counts, width, label='Total Turns')
            plt.bar([i + width for i in x], valid_counts, width, label='Valid Format')
            
            plt.xlabel('Conversation')
            plt.ylabel('Count')
            plt.title('Multi-turn Conversation Performance')
            plt.xticks([i + width/2 for i in x], conv_names)
            plt.legend()
            
            # Save visualization
            conv_viz_file = self.results_dir / f"conv_viz_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(conv_viz_file)
            print(f"Conversation visualization saved to {conv_viz_file}")

def main():
    """Main function to run the model test."""
    print("🚀 Starting React-Respond-Reflect model test...")
    
    # Initialize tester
    tester = RRRModelTester(
        model_path="rrr_model",  # Path to the model directory
        max_new_tokens=2048,     # Maximum tokens to generate
    )
    
    # Run tests
    results = tester.run_tests()
    
    # Generate report
    tester.generate_report(results)
    
    print("✅ Test completed!")

if __name__ == "__main__":
    main() 