from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, AsyncGenerator
import uvicorn
import time
import os
import torch
import asyncio
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import re
import json

# Initialize FastAPI app
app = FastAPI()

# Define model path
MODEL_PATH = os.environ.get("MODEL_PATH", "./rrr_model_merged")

# Load model and tokenizer
print(f"Loading model from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class GenerationRequest(BaseModel):
    """Model for generation request."""
    prompt: str = Field(..., description="Input text prompt for generation")
    max_new_tokens: Optional[int] = Field(
        128, description="Maximum number of new tokens to generate"
    )
    temperature: Optional[float] = Field(
        0.7, description="Sampling temperature (0.0 = deterministic, 1.0 = more random)"
    )
    top_p: Optional[float] = Field(
        0.9, description="Nucleus sampling parameter (1.0 = no effect)"
    )
    top_k: Optional[int] = Field(
        40, description="Top-k sampling parameter (0 = no effect)"
    )
    repetition_penalty: Optional[float] = Field(
        1.1, description="Penalty for repetition (1.0 = no penalty)"
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream the response"
    )

class GenerationMetadata(BaseModel):
    """Model for generation metadata."""
    input_tokens: int
    output_tokens: int
    inference_time_seconds: float
    tokens_per_second: float

class GenerationResponse(BaseModel):
    """Model for generation response."""
    generated_text: str
    metadata: GenerationMetadata
    error: Optional[str] = None

class StreamingChunk(BaseModel):
    """Model for streaming response chunks."""
    token: str
    finished: bool = False
    metadata: Optional[GenerationMetadata] = None

def validate_rrr_format(text: str) -> Dict[str, Any]:
    """
    Validate that responses follow React-Respond-Reflect format.
    
    Args:
        text: The text to validate
            
    Returns:
        Dict with validation results
    """
    # Extract assistant response
    assistant_text = text
    
    # Check for all three tags
    react_match = re.search(r'<react>\s*(.*?)\s*</react>', assistant_text, re.DOTALL)
    respond_match = re.search(r'<respond>\s*(.*?)\s*</respond>', assistant_text, re.DOTALL)
    reflect_match = re.search(r'<reflect>\s*(.*?)\s*</reflect>', assistant_text, re.DOTALL)
    
    has_all_tags = all([react_match, respond_match, reflect_match])
    
    # Check order if all tags are present
    correct_order = False
    if has_all_tags:
        react_pos = assistant_text.find('<react>')
        respond_pos = assistant_text.find('<respond>')
        reflect_pos = assistant_text.find('<reflect>')
        correct_order = (react_pos < respond_pos < reflect_pos)
    
    # Check content in each section
    tag_content = {
        "react": bool(react_match and len(react_match.group(1).strip()) > 10),
        "respond": bool(respond_match and len(respond_match.group(1).strip()) > 20),
        "reflect": bool(reflect_match and len(reflect_match.group(1).strip()) > 10),
    }
    
    return {
        "valid": has_all_tags and correct_order,
        "has_all_tags": has_all_tags,
        "correct_order": correct_order,
        "tag_content": tag_content
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

async def generate_stream(request: GenerationRequest) -> AsyncGenerator[str, None]:
    """Generate text in a streaming fashion."""
    start_time = time.time()
    
    # Format the prompt using the chat template
    messages = [
        {"role": "system", "content": "You are an empathetic AI assistant. Always respond in the React-Respond-Reflect format using <react>, <respond>, and <reflect> tags in that order."},
        {"role": "user", "content": request.prompt}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_tokens = input_ids.shape[1]
    
    # Create a streamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, timeout=10.0)
    
    # Generate in a separate thread
    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the output
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        response_chunk = StreamingChunk(token=new_text, finished=False)
        yield json.dumps(response_chunk.dict()) + "\n"
    
    # Calculate metadata
    inference_time = time.time() - start_time
    output_tokens = len(generated_text.split())
    tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0
    
    # Validate RRR format
    format_validation = validate_rrr_format(generated_text)
    if not format_validation["valid"]:
        print(f"Warning: Generated text does not follow RRR format: {format_validation}")
    
    # Send final chunk with metadata
    final_chunk = StreamingChunk(
        token="",
        finished=True,
        metadata=GenerationMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            inference_time_seconds=inference_time,
            tokens_per_second=tokens_per_second
        )
    )
    yield json.dumps(final_chunk.dict()) + "\n"

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate text based on input prompt."""
    # If streaming is requested, use the streaming endpoint
    if request.stream:
        return StreamingResponse(generate_stream(request), media_type="text/event-stream")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Format the prompt using the chat template
        messages = [
            {"role": "system", "content": "You are an empathetic AI assistant. Always respond in the React-Respond-Reflect format using <react>, <respond>, and <reflect> tags in that order."},
            {"role": "user", "content": request.prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        input_tokens = input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the output
        generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Calculate metadata
        inference_time = time.time() - start_time
        output_tokens = len(output[0]) - input_tokens
        tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0
        
        # Validate RRR format
        format_validation = validate_rrr_format(generated_text)
        if not format_validation["valid"]:
            print(f"Warning: Generated text does not follow RRR format: {format_validation}")
        
        return GenerationResponse(
            generated_text=generated_text,
            metadata=GenerationMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                inference_time_seconds=inference_time,
                tokens_per_second=tokens_per_second
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 