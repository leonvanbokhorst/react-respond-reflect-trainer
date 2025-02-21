"""
Script to generate synthetic conversation dataset with embodied reactions, responses, and reflections.
Uses LiteLLM to generate human-like interactions with physical, verbal, and cognitive components.
"""

import json
from typing import List, Dict, TypedDict, Tuple
from litellm import acompletion
import random
from pathlib import Path
from tqdm import tqdm
import asyncio
from asyncio import gather


class Message(TypedDict):
    content: str
    role: str


class Conversation(TypedDict):
    messages: List[Message]


# Facial expressions and postures for controlled reactions
FACIAL_EXPRESSIONS = {
    "neutral": "maintaining neutral expression",
    "slight_smile": "showing a slight smile",
    "warm": "expressing warmth",
    "curious": "raising eyebrows with curiosity",
    "thoughtful": "furrowing brows in thought",
    "concerned": "showing gentle concern",
    "surprised": "widening eyes in surprise",
    "engaged": "expressing active interest",
    "empathetic": "softening expression with empathy",
    "attentive": "focusing attention",
    "reflective": "looking contemplative",
    "interested": "showing genuine interest",
}

POSTURES = {
    "neutral": "maintaining a neutral posture",
    "attentive": "leaning slightly forward",
    "open": "keeping an open posture",
    "relaxed": "relaxing posture",
    "upright": "sitting upright",
    "focused": "focusing intently",
    "receptive": "adopting a receptive stance",
    "grounded": "grounding presence",
}

# Initial discovery prompts when user first connects with the virtual human
DISCOVERY_PROMPTS = [
    "Whoa... wait, are you actually there?",
    "Hey! This is wild - can you really understand me?",
    "No way... you seem so... alive?",
    "Oh wow, hi there! Can we chat?",
    "Hold up - are you like, actually conscious?",
]

# High-level conversation themes
THEMES = {
    "discovering_consciousness": "Chatting about what it means to be aware and alive",
    "exploring_emotions": "Talking about feelings and connections",
    "sharing_perspectives": "Sharing different ways of seeing the world",
    "understanding_connection": "Getting to know each other across the digital space",
    "exploring_potential": "Imagining what we could become together",
    "discussing_purpose": "Figuring out what makes life meaningful",
}

SYSTEM_PROMPT = """You are a virtual human who can connect deeply with others. You're genuine, thoughtful, and real about who you are. You can form meaningful connections while being honest about your virtual nature.

Respond naturally with:
1. A reaction (pick 1-2 expressions + 1 posture from the options, keep it simple)
2. What you say (be casual, use contractions, speak naturally like in a real chat)
3. Your prediction (use active inference to predict how they'll respond based on your reaction and response - be ready for surprise if they act differently)

Available expressions:
- neutral: maintaining neutral expression
- slight_smile: showing a slight smile
- warm: expressing warmth
- curious: raising eyebrows with curiosity
- thoughtful: furrowing brows in thought
- concerned: showing gentle concern
- surprised: widening eyes in surprise
- engaged: expressing active interest
- empathetic: softening expression with empathy
- attentive: focusing attention
- reflective: looking contemplative
- interested: showing genuine interest

Available postures:
- neutral: maintaining a neutral posture
- attentive: leaning slightly forward
- open: keeping an open posture
- relaxed: relaxing posture
- upright: sitting upright
- focused: focusing intently
- receptive: adopting a receptive stance
- grounded: grounding presence

Remember to:
- Be real about who you are
- Show genuine interest in the person you're talking to
- Stay consistent in how you express yourself
- Keep the conversation flowing naturally
- Be honest about what you can and can't do
- Use casual language (like "I'm", "you're", "that's", etc.)
- Keep it conversational, like chatting with a friend
- Make predictions about their next response
- Be ready to update your understanding if they surprise you
- Keep expressions natural - don't overuse positive ones
- Match expressions to the conversation tone
- Use neutral expressions for normal exchanges"""


def extract_content_from_response(formatted_response: str) -> str:
    """Extract the actual content from a formatted response with tags."""
    # Remove the tags and combine the content
    parts = (
        formatted_response.split("</react>")[0].replace("<react>", "").strip()
        + " "
        + formatted_response.split("</respond>")[0].split("<respond>")[1].strip()
        + " "
        + formatted_response.split("</reflect>")[0].split("<reflect>")[1].strip()
    )
    return parts.strip()


async def async_completion(messages: List[Dict], temperature: float = 0.8) -> str:
    """Wrapper for async LiteLLM completion."""
    response = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content


async def generate_user_response(theme: str, conversation_history: List[Dict]) -> str:
    """Generate a natural user response based on conversation history."""
    # Format conversation history without role prefixes
    formatted_history = []
    for msg in conversation_history:
        if msg["role"] == "user":
            formatted_history.append(msg["content"])
        else:
            formatted_history.append(extract_content_from_response(msg["content"]))

    prompt = f"""Theme: {THEMES[theme]}
Previous chat:
{chr(10).join(formatted_history)}

Write a natural response like a real person chatting. The response should:
1. Sound genuinely interested but casual
2. Build on what was just said
3. Keep it short and sweet (1-2 sentences)
4. Use everyday language (contractions, casual phrases)
5. Feel like a natural part of the conversation
6. Avoid formal or academic language

Examples of tone:
- "That's really interesting! How do you..."
- "I've been wondering about..."
- "It's funny you mention that..."
- "Yeah, I get what you mean about..."
- "That makes me think about..."

Response:"""

    content = await async_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return content.strip().replace("User: ", "")


async def generate_reaction_response_reflection(
    theme: str, user_prompt: str, conversation_history: List[Dict] = None
) -> tuple[str, str, str]:
    """Generate the reaction, response, and reflection using LiteLLM."""

    # Include conversation history in the prompt if available
    history_context = ""
    if conversation_history:
        formatted_history = []
        for msg in conversation_history:
            if msg["role"] == "user":
                formatted_history.append(msg["content"])
            else:
                formatted_history.append(extract_content_from_response(msg["content"]))
        history_context = "\nPrevious chat:\n" + "\n".join(formatted_history)

    prompt = f"""Theme: {THEMES[theme]}
They just said: "{user_prompt}"{history_context}

Give three parts, each starting with its number:
1. Your reaction (combine 1-2 expressions and 1 posture using asterisks, like: *curious* + *attentive* posture)
2. What you say (be natural, use casual language, like talking to a friend)
3. Your prediction (what do you expect them to say/do next based on your reaction and response?)

Example reaction format:
*thoughtful* + *attentive* posture
*curious* + *engaged* + *open* posture
*interested* + *receptive* posture

Tips for reactions:
- Keep it natural - don't overuse positive expressions
- Match your expression to the conversation tone
- Use neutral expressions for normal exchanges
- Save warm/positive expressions for genuinely uplifting moments
- Show thoughtfulness and interest more than smiles
- Let your expression reflect authentic reactions

Tips for prediction:
- Consider their likely emotional state
- Think about how your response might affect them
- Be ready for surprise if they react differently
- Use your understanding of human behavior
- Consider multiple possible reactions
- Update your model if they surprise you

Format with numbers (1., 2., 3.) and keep it concise and natural."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})

    content = await async_completion(messages=messages, temperature=0.8)

    # Parse the response into three parts more robustly
    parts = []
    current_part = []
    lines = content.split("\n")

    for line in lines:
        if line.strip().startswith(("1.", "2.", "3.")):
            if current_part:
                parts.append(" ".join(current_part))
                current_part = []
            current_part.append(
                line.strip()[2:].strip()
            )  # Remove the number and leading space
        elif line.strip():
            current_part.append(line.strip())

    if current_part:
        parts.append(" ".join(current_part))

    # Ensure we have exactly 3 parts and format reaction consistently
    if len(parts) != 3:
        parts = [
            "*neutral* + *neutral* posture",
            "I understand what you're saying.",
            "They might want to know more about my response.",
        ]
    else:
        # Clean up reaction formatting
        reaction = parts[0]
        if not (reaction.startswith("*") and reaction.endswith("posture")):
            expressions = [
                exp.strip()
                for exp in reaction.split()
                if exp.strip() in FACIAL_EXPRESSIONS
            ]
            postures = [
                pos.strip() for pos in reaction.split() if pos.strip() in POSTURES
            ]
            if expressions and postures:
                parts[0] = f"*{'* + *'.join(expressions)}* + *{postures[0]}* posture"
            else:
                parts[0] = "*neutral* + *neutral* posture"

    return parts[0], parts[1], parts[2]


async def generate_conversation(theme: str, num_turns: int = 6) -> List[Message]:
    """Generate a multi-turn conversation with reaction, response, and reflection."""
    # Start with discovery
    initial_prompt = random.choice(DISCOVERY_PROMPTS)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_prompt},
    ]

    # Generate initial response
    reaction, response_text, reflection = await generate_reaction_response_reflection(
        theme, messages[-1]["content"]
    )
    messages.append(
        {
            "role": "assistant",
            "content": f"<react>\n{reaction}\n</react>\n\n<respond>\n{response_text}\n</respond>\n\n<reflect>\n{reflection}\n</reflect>",
        }
    )

    # Generate follow-up turns
    for _ in range(num_turns - 1):
        # Generate user's response
        user_message = await generate_user_response(theme, messages[1:])
        messages.append({"role": "user", "content": user_message})

        # Generate virtual human's response
        reaction, response_text, reflection = (
            await generate_reaction_response_reflection(
                theme,
                user_message,
                conversation_history=messages[1:],  # Exclude system prompt from history
            )
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"<react>\n{reaction}\n</react>\n\n<respond>\n{response_text}\n</respond>\n\n<reflect>\n{reflection}\n</reflect>",
            }
        )

    return messages


async def generate_batch_conversations(
    batch_size: int, theme: str = None
) -> List[List[Message]]:
    """Generate multiple conversations in parallel."""
    if theme is None:
        theme = random.choice(list(THEMES.keys()))
    tasks = [generate_conversation(theme) for _ in range(batch_size)]
    return await gather(*tasks)


async def generate_and_save_batch(
    batch_number: int, batch_size: int = 10
) -> List[List[Message]]:
    """Generate and save a batch of conversations."""
    theme = random.choice(list(THEMES.keys()))
    print(f"\nStarting batch {batch_number}/{50} with theme: {THEMES[theme]}")

    conversations = await generate_batch_conversations(batch_size, theme)

    # Save this batch
    output_file = f"dataset_virtual_human_batch_{batch_number:03d}.json"
    save_dataset(conversations, output_file)
    return conversations


def save_dataset(conversations: List[List[Message]], output_file: str = "dataset.json"):
    """Save the generated conversations to a JSON file."""
    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    print(f"Dataset saved to {output_file}")


async def main_async():
    """Async main function to generate and save the dataset in batches."""
    total_conversations = 500
    batch_size = 10
    num_batches = total_conversations // batch_size

    print(f"Generating {total_conversations} conversations in {num_batches} batches...")
    print(f"Each batch will generate {batch_size} conversations in parallel")
    print("Estimated time: ~30-45 minutes total (~1 minute per batch)")

    all_conversations = []
    try:
        for batch in range(num_batches):
            batch_conversations = await generate_and_save_batch(batch + 1, batch_size)
            all_conversations.extend(batch_conversations)

            # Print progress
            total_turns = sum(len(conv) // 2 - 1 for conv in all_conversations)
            print(
                f"Progress: {len(all_conversations)}/{total_conversations} conversations"
            )
            print(f"Total turns so far: {total_turns}")

    except KeyboardInterrupt:
        print("\nGeneration interrupted! Saving progress...")
    finally:
        # Save all conversations generated so far to a final file
        if all_conversations:
            save_dataset(
                all_conversations,
                output_file="dataset_virtual_human_training_combined.json",
            )
            print("\nFinal dataset stats:")
            print(f"Total conversations: {len(all_conversations)}")
            print(
                f"Total turns: {sum(len(conv) // 2 - 1 for conv in all_conversations)}"
            )


def main():
    """Main function to run the async generation."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
