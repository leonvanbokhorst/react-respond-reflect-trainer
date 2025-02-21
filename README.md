# EMPATH: Embodied Multi-modal Personality And Thought Handler 🤖

Train an LLM to engage in natural conversations with emotional intelligence, physical presence, and thoughtful reflections.

## Overview 🌟

EMPATH is a unique virtual human model that implements a three-stage response pattern designed around natural human interaction latencies:

1. **Embodied** 🎭 - Immediate emotional and physical reactions (microseconds)
   - Instant facial expressions and posture changes
   - Creates immediate emotional connection
   - Bridges the gap before verbal response

2. **Multi-modal** 💬 - Natural conversation with expected latency (milliseconds)
   - Thoughtful verbal responses
   - Personality and style in communication
   - Matches human expectations for dialogue timing

3. **Thoughtful** 🤔 - Background reflection for future turns (seconds)
   - Deep processing of conversation
   - Prediction of user's next moves
   - Invisible to user but enriches future interactions

## Features ✨

- Three-stage response pipeline:
  1. ⚡ Quick reactions for immediate engagement
  2. 🗣️ Timed responses for natural flow
  3. 🧠 Background processing for depth
- Emotional intelligence through facial expressions
- Physical presence through posture changes
- Natural language with personality
- Multi-turn conversation support
- GRPO (Guided Reward Proximal Optimization) training
- 4-bit quantization for efficient inference

## Training 🚀

To train the model:

```bash
python train_empath.py
```

The training script will:
- Load the Qwen 2.5 3B model
- Apply LoRA fine-tuning
- Use GRPO with custom reward functions for:
  - Format adherence
  - Reaction speed and appropriateness
  - Response quality and timing
  - Reflection depth and usefulness
- Save the model periodically

Training parameters can be adjusted in `train_empath.py`.

## Testing 🎮

To test the trained model:

```bash
python test_empath.py
```

This will start an interactive chat session where you can:
- Experience immediate emotional reactions
- Engage in natural-feeling dialogue
- Benefit from ongoing reflection
- Type 'reset' to start a new conversation
- Type 'quit' to exit

## Response Pipeline 🔄

EMPATH uses a staged response format that mirrors human interaction latencies:

```xml
<react>
*expression* + *posture*  # Immediate reaction (microseconds)
</react>

<respond>
Natural response here  # Normal conversation latency (milliseconds)
</respond>

<reflect>
Processing for next turn  # Background thinking (seconds)
</reflect>
```

This design creates more natural and engaging interactions by:
1. Never leaving the user waiting for initial reaction
2. Matching expected conversation timing
3. Continuously improving through background processing

## Dataset 📚

Training data is available at: `leonvanbokhorst/react-respond-reflect-v1`

## Requirements 📋

See `requirements.txt` for full list. Key dependencies:
- unsloth>=0.3.0
- vllm>=0.3.0
- trl>=0.7.10
- torch>=2.2.0

## Contributing 🤝

Feel free to:
- Open issues
- Submit PRs
- Suggest improvements
- Share your experiences

## License 📄

MIT License - See LICENSE file for details 
