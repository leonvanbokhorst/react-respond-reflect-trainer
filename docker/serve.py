"""
Serve a React-Respond-Reflect model using vLLM.

This script sets up a vLLM server for the fine-tuned RRR model,
providing both OpenAI-compatible and direct HTTP endpoints.

Features:
- OpenAI-compatible API endpoint
- Custom chat template for RRR format
- Configurable model parameters
- Health check endpoint
- Proper error handling

Environment Variables:
    MODEL_PATH (str): Path to the model directory (default: /app/rrr_model)
    PORT (int): Port to run the server on (default: 9999)
    MAX_MODEL_LEN (int): Maximum sequence length (default: 2048)
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    UsageInfo,
)
from vllm.utils import random_uuid

# Get environment variables with defaults
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/rrr_model")
PORT = int(os.environ.get("PORT", "9999"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))

# Create FastAPI app
app = FastAPI(
    title="React-Respond-Reflect API",
    description="API for serving React-Respond-Reflect models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom request model for RRR chat
class RRRChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(
        ..., description="List of messages in the conversation"
    )
    temperature: Optional[float] = Field(
        0.7, description="Sampling temperature between 0 and 2"
    )
    max_tokens: Optional[int] = Field(
        512, description="Maximum number of tokens to generate"
    )
    stream: Optional[bool] = Field(False, description="Whether to stream the response")


# Initialize the LLM engine
@app.on_event("startup")
async def startup_event():
    global engine

    # Configure engine arguments
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",  # Use bfloat16 for efficiency
        tensor_parallel_size=1,  # Adjust based on available GPUs
        trust_remote_code=True,  # Required for some models
        chat_template_file="/app/chat_template.jinja",  # Custom chat template
    )

    # Create the engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"🚀 vLLM engine initialized with model: {MODEL_PATH}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_PATH}


# OpenAI-compatible chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens or 512,
        stop=request.stop or None,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
    )

    # Process the request
    results_generator = engine.generate(
        prompt=None,
        sampling_params=sampling_params,
        request_id=random_uuid(),
        prompt_token_ids=None,
        messages=request.messages,
    )

    # Get the results
    final_output = None
    async for output in results_generator:
        final_output = output

    if final_output is None:
        raise HTTPException(status_code=500, detail="Failed to generate response")

    # Format the response
    choices = []
    for i, output in enumerate(final_output.outputs):
        choice = ChatCompletionResponseChoice(
            index=i,
            message=ChatMessage(
                role="assistant",
                content=output.text,
            ),
            finish_reason=output.finish_reason,
        )
        choices.append(choice)

    # Calculate token usage
    prompt_tokens = final_output.prompt_token_ids.shape[0]
    completion_tokens = sum(len(output.token_ids) for output in final_output.outputs)

    # Create the response
    response = ChatCompletionResponse(
        id=random_uuid(),
        object="chat.completion",
        created=int(final_output.timestamp),
        model=MODEL_PATH,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


# Custom RRR chat endpoint
@app.post("/rrr/chat")
async def rrr_chat(request: RRRChatRequest):
    """Custom endpoint for React-Respond-Reflect chat."""

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    # Process the request
    results_generator = engine.generate(
        prompt=None,
        sampling_params=sampling_params,
        request_id=random_uuid(),
        prompt_token_ids=None,
        messages=request.messages,
    )

    # Get the results
    final_output = None
    async for output in results_generator:
        final_output = output

    if final_output is None:
        raise HTTPException(status_code=500, detail="Failed to generate response")

    # Extract the RRR components using regex
    import re

    response_text = final_output.outputs[0].text

    # Extract react, respond, reflect components
    react_match = re.search(r"<react>\*(.*?)\*</react>", response_text)
    respond_match = re.search(r"<respond>(.*?)</respond>", response_text)
    reflect_match = re.search(r"<reflect>(.*?)</reflect>", response_text)

    # Format the response
    response = {
        "id": random_uuid(),
        "created": int(final_output.timestamp),
        "model": MODEL_PATH,
        "content": response_text,
        "components": {
            "react": react_match.group(1) if react_match else None,
            "respond": respond_match.group(1) if respond_match else None,
            "reflect": reflect_match.group(1) if reflect_match else None,
        },
        "usage": {
            "prompt_tokens": final_output.prompt_token_ids.shape[0],
            "completion_tokens": len(final_output.outputs[0].token_ids),
            "total_tokens": final_output.prompt_token_ids.shape[0]
            + len(final_output.outputs[0].token_ids),
        },
    }

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
