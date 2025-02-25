# React-Respond-Reflect Framework 🎭

This repository contains both a curated dialogue dataset and the tools used to generate it. The project aims to improve AI-human interactions through structured, empathetic conversation patterns.

## Project Components 📦

1. **Dataset**: A collection of dialogues demonstrating the React-Respond-Reflect framework
2. **Generation Tools**: Python scripts for creating and processing dialogue data


## Dataset Description 📊

### Overview
The dataset contains dialogues between users and a virtual human, where each response follows a three-part structure:
- **React**: Physical/emotional reactions expressed through actions and body language
- **Respond**: The actual verbal response to the user
- **Reflect**: Internal thoughts and analysis of the conversation

### Format
```json
{
    "conversation_id": "unique_id",
    "messages": [
        {
            "role": "user",
            "content": "user message"
        },
        {
            "role": "assistant",
            "content": "virtual human response with react/respond/reflect tags"
        }
    ],
    "num_turns": "number of back-and-forth exchanges"
}
```

### Topics Covered 📝
- Work-related stress and challenges
- Personal development and growth
- Technical learning and coding
- Time management and productivity
- Interpersonal relationships
- Mental health and wellbeing

## Generation Tools 🛠️

### Scripts
1. `seed_dialogues_generate_dataset.py`
   - Generates dialogues using GPT-4-mini
   - Batch processing with progress tracking
   - Temperature-based randomization
   - Automatic validation

2. `seed_dialogues_convert_to_hf.py`
   - Converts to HuggingFace format
   - Generates dataset statistics
   - Handles dataset publishing

3. `seed_dialogues_validate_tags.py`
   - Validates XML-style tags
   - Fixes formatting issues
   - Provides detailed reporting

4. `seed_dialogues_save_curated.py`
   - Handles manual curation workflow
   - Creates automatic backups
   - Preserves dialogue structure

5. `benchmark_rrr.py`
   - Evaluates model performance against baseline
   - Calculates format compliance and quality metrics
   - Generates visualizations and detailed reports
   - Computes NLP metrics using reference responses
   - Analyzes performance across different prompt categories

### Setup & Usage 🚀

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Add your API keys:
# - OPENAI_API_KEY: For dialogue generation
# - HF_TOKEN: For HuggingFace upload
```

3. Run tools:
```bash
# Generate dialogues
python seed_dialogues_generate_dataset.py

# Convert to HuggingFace format
python seed_dialogues_convert_to_hf.py

# Validate tags
python seed_dialogues_validate_tags.py

# Save curated dialogues
python seed_dialogues_save_curated.py

# Run benchmark evaluation
python benchmark_rrr.py
```

### Directory Structure 📁
```
.
├── curated_seed_dialogs/     # Curated examples
├── dialogs_to_curate/        # Pending curation
├── seed_dialogues_*.json     # Generated batches
└── requirements.txt          # Dependencies
```

## Fine-tuned Model 🤖

We've fine-tuned a Mistral-7B model to follow the React-Respond-Reflect framework, creating a conversational AI that provides structured, empathetic responses.

### Model Features

- **Format Adherence**: Consistently follows the three-part structure
- **Reasoning Quality**: Demonstrates thoughtful internal processing
- **Response Quality**: Provides helpful, contextually appropriate answers
- **Reflection Depth**: Shows self-awareness and conversation analysis
- **Fast Response**: Generates complete responses in ~3 seconds

### Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="rrr_model",  # Local path or HuggingFace repo
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    map_eos_token=True,
)

# Prepare for inference
FastLanguageModel.for_inference(model)

# Generate a response
messages = [{"role": "user", "content": "I'm feeling anxious about my job interview tomorrow. Any advice?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=2048,
    temperature=0.7,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

## Using the Dataset 💡

### Loading
```python
from datasets import load_dataset
dataset = load_dataset("leonvanbokhorst/react-respond-reflect-dialogues-v2")
```

### Applications
- Training conversational AI models
- Studying empathetic response patterns
- Analyzing structured dialogue frameworks
- Developing emotional intelligence in chatbots

## Benchmark Results 📊

We've conducted comprehensive benchmarking of our fine-tuned React-Respond-Reflect model against a baseline model (Mistral-7B). The results demonstrate significant improvements in format compliance, response quality, and processing speed.

### Performance Metrics

| Metric | Fine-tuned Model | Baseline Model |
|--------|-----------------|---------------|
| Format Compliance | 100.0% | 0.0% |
| Reasoning Quality | 7.6/10 | 0.0/10 |
| Response Quality | 7.4/10 | 0.0/10 |
| Reflection Depth | 7.2/10 | 0.0/10 |
| Response Time | 3.0s | 47.3s |

### NLP Metrics

| Metric | Score |
|--------|-------|
| BLEU | 0.028 |
| ROUGE-1 | 0.257 |
| ROUGE-2 | 0.054 |
| ROUGE-L | 0.194 |
| METEOR | 0.174 |
| BERTScore F1 | 0.189 |
| Semantic Similarity | 0.379 |

### Visualizations

#### Quality Metrics Comparison
![Quality Metrics Comparison](benchmark_results/quality_radar.png)

#### Response Time Comparison
![Response Time Comparison](benchmark_results/response_time.png)

#### NLP Metrics
![NLP Metrics Comparison](benchmark_results/nlp_metrics_comparison.png)

#### Category Analysis
![Category Analysis](benchmark_results/category_analysis.png)

### Benchmark Methodology

The benchmark evaluates:
1. **Format compliance**: Adherence to the React-Respond-Reflect structure
2. **Quality metrics**: Reasoning depth, response helpfulness, and reflection insight
3. **Response time**: Generation speed for complete responses
4. **NLP metrics**: Similarity to reference responses using BLEU, ROUGE, METEOR, and BERTScore
5. **Category performance**: Performance across emotional support, practical advice, philosophical, and adversarial prompts

The benchmark uses a combination of:
- Automated format validation
- GPT-4o-mini for quality evaluation
- Reference-based NLP metrics
- Stratified sampling across prompt categories

## Contributing 🤝

1. Follow PEP 8 style guide
2. Use type hints (PEP 484)
3. Add Google-style docstrings
4. Run validation before committing

## Citation 📚

```bibtex
@dataset{react_respond_reflect_dialogues,
  author = {van Bokhorst, Leon},
  title = {React-Respond-Reflect Dialogues Dataset},
  year = {2025},
  publisher = {HuggingFace},
  version = {2.0},
  url = {https://huggingface.co/datasets/leonvanbokhorst/react-respond-reflect-dialogues-v2}
}
```


