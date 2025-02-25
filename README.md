# React-Respond-Reflect Framework 🎭

This repository contains both a curated dialogue dataset and the tools used to generate it. The project aims to improve AI-human interactions through structured, empathetic conversation patterns.

## Project Components 📦

1. **Dataset**: A collection of dialogues demonstrating the React-Respond-Reflect framework
2. **Generation Tools**: Python scripts for creating and processing dialogue data
3. **Training Pipeline**: Scripts for fine-tuning models on the RRR framework
4. **Deployment Tools**: Docker and vLLM integration for serving trained models

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

## Training Pipeline 🚂

### Training Script

The `rrr_train.py` script provides a complete pipeline for fine-tuning models on the RRR framework:

- Uses Unsloth for efficient training
- Supports LoRA fine-tuning
- Implements ChatML format
- Validates RRR format in outputs
- Optimized for consumer GPUs

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

# Train the model
python rrr_train.py
```

## Deployment with Docker and vLLM 🐳

This project includes Docker and vLLM integration for easy deployment of trained models.

### Export Model for vLLM

After training, export your model for vLLM deployment:

```bash
python rrr_export_for_vllm.py --input_dir ./rrr_model --output_dir ./rrr_model_vllm
```

### Docker Deployment

Deploy your model with Docker:

```bash
# Build and start the container
docker-compose up --build

# Or run with specific GPU
docker-compose up --build -d
```

The server will be available at http://localhost:8000 with the following endpoints:

- `/health`: Health check endpoint
- `/v1/chat/completions`: OpenAI-compatible chat completions API
- `/rrr/chat`: Custom RRR-specific endpoint with component extraction

### Testing the Deployed Model

Use the included test client to interact with your deployed model:

```bash
python docker/test_client.py --endpoint http://localhost:8000 --api rrr
```

### Directory Structure 📁

```
.
├── curated_seed_dialogs/     # Curated examples
├── dialogs_to_curate/        # Pending curation
├── docker/                   # Docker deployment files
├── rrr_model/                # Trained model output
├── rrr_model_vllm/           # Exported model for vLLM
├── seed_dialogues_*.json     # Generated batches
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── rrr_export_for_vllm.py    # Model export script
├── rrr_train.py              # Training script
└── requirements.txt          # Dependencies
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
  year = {2024},
  publisher = {HuggingFace},
  version = {2.0},
  url = {https://huggingface.co/datasets/leonvanbokhorst/react-respond-reflect-dialogues-v2}
}
```

## License 📜

MIT License - See LICENSE file for details
