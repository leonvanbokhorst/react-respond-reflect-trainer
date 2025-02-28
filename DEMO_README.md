# React-Respond-Reflect Demo for Mac M3 🎭

This is a simplified demo version of the React-Respond-Reflect framework, optimized for running on Apple Silicon (M3) MacBooks.

## Quick Start 🚀

1. Make the setup script executable:
   ```bash
   chmod +x setup_mac_demo.sh
   ```

2. Run the setup script:
   ```bash
   ./setup_mac_demo.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv_mac/bin/activate
   ```

4. Run the demo:
   ```bash
   python rrr_demo.py
   ```

## Model Options 🤗

By default, the demo will use the model from Hugging Face Hub. You have several options:

- Use the default HF model:
  ```bash
  python rrr_demo.py
  ```

- Use a specific HF model:
  ```bash
  python rrr_demo.py --model_path username/repo-name
  ```

- Use a local model:
  ```bash
  python rrr_demo.py --model_path ./rrr_model
  ```

## Device Options ⚙️

- Run on Apple Silicon (MPS) - default and faster:
  ```bash
  python rrr_demo.py --device mps
  ```

- Run on CPU only (slower but more stable):
  ```bash
  python rrr_demo.py --device cpu
  ```

## Uploading to Hugging Face 📤

If you have a local model and want to upload it to Hugging Face:

1. Make sure you have a Hugging Face account and API token
2. Run the upload script:
   ```bash
   python upload_to_hf.py --repo_name your-username/your-repo-name
   ```

Or use the interactive option in the setup script.

## Troubleshooting 🔧

### Common Issues

1. **Memory Errors**: If you encounter memory errors, try:
   - Using the CPU option: `python rrr_demo.py --device cpu`
   - Closing other applications to free up memory

2. **Slow Generation**: The first run will be slower as models are downloaded and cached.

3. **Missing Dependencies**: If you encounter missing dependencies, try:
   ```bash
   pip install -r requirements_mac.txt
   ```

4. **Model Not Found**: If using a local model, ensure the files are in the correct location:
   ```
   rrr_model/
   ├── adapter_model.safetensors
   ├── adapter_config.json
   ├── tokenizer.json
   └── ...
   ```

## How It Works 🧠

The demo uses:

1. **Base Model**: Mistral-7B as the foundation
2. **LoRA Adapter**: Fine-tuned weights for the React-Respond-Reflect framework
3. **Apple MPS**: Metal Performance Shaders for hardware acceleration on Apple Silicon

## Example Prompts 💬

Try these prompts to see the model in action:

- "I'm feeling anxious about my job interview tomorrow. Any advice?"
- "How can I improve my focus when working from home?"
- "I'm struggling with imposter syndrome in my new role."
- "What are some strategies for managing my time better?"
- "I feel overwhelmed by all the tasks I need to complete."

## Format Explanation 📝

The model responses follow the React-Respond-Reflect format:

- **React**: Physical/emotional reactions expressed through actions and body language
- **Respond**: The actual verbal response to the user
- **Reflect**: Internal thoughts and analysis of the conversation 