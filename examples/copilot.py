# This is an example of a http steaming server support Github Copilot VSCode extension
# The server depends on fastapi/uvicorn/huggingface-hub (for auto download the model files), to run the server:
# 1. `pip install uvicorn fastapi huggingface-hub`
# 2. `uvicorn copilot:app --reload --host 0.0.0.0 --port 9999`
# 3. Configure VSCode copilot extension (in VSCode's settings.json):
# ```json
# "github.copilot.advanced": {
#     "debug.overrideEngine": "engine", # can be any string.
#     "debug.testOverrideProxyUrl": "http://localhost:9999",
#     "debug.overrideProxyUrl": "http://localhost:9999"
# }
# ```

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from os import times
import logging
import json

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from huggingface_hub import snapshot_download
from typing import List, Optional
from pydantic import BaseModel

log = logging.getLogger("uvicorn")
log.setLevel("DEBUG")
app = FastAPI()

# Find one here https://huggingface.co/turboderp
MODEL_HG_REPO_ID = "turboderp/CodeLlama-34B-instruct-3.0bpw-h6-exl2"


@app.on_event("startup")
async def startup_event():
    """_summary_
    Starts up the server, setting log level, downloading the default model if necessary.

    Edited from https://github.com/chenhunghan/ialacol/blob/main/main.py
    """
    log.info("Starting up...")
    log.info(
        "Downloading repo %s to %s/models",
        MODEL_HG_REPO_ID,
        os.getcwd(),
    )
    snapshot_download(
        repo_id=MODEL_HG_REPO_ID,
        cache_dir=f"{os.getcwd()}/models/.cache",
        local_dir=f"{os.getcwd()}/models",
        resume_download=True,
    )
    log.debug("Creating generator instance...")
    model_directory = f"{os.getcwd()}/models"
    config = ExLlamaV2Config()
    config.model_dir = model_directory
    config.prepare()
    tokenizer = ExLlamaV2Tokenizer(config)
    log.debug("Creating tokenizer instance...")
    model = ExLlamaV2(config)
    log.debug("Loading model...")
    model.load([16, 24])
    log.debug("Creating cache instance...")
    cache = ExLlamaV2Cache(model)

    log.debug("Creating generator instance...")
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    # Ensure CUDA is initialized
    log.debug("Warming up generator instance...")
    generator.warmup()
    app.state.generator = generator
    app.state.tokenizer = tokenizer
    log.debug("Generator instance created.")


class CompletionRequestBody(BaseModel):
    """_summary_
    from from https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/server/app.py
    """

    prompt: str = ""
    max_tokens: Optional[int] = 99999
    temperature: Optional[float] = 0.85
    top_p: Optional[float] = 0.8
    stop: Optional[List[str] | str] = ["\ndef ", "\nclass ", "\nif ", "\n\n#"]
    stream: bool = True
    model: str = ""
    top_k: Optional[int] = 50

    repetition_penalty: Optional[float] = 15

    class Config:
        arbitrary_types_allowed = True


@app.post("/v1/engines/{engine}/completions")
async def engine_completions(
    # Can't use body as FastAPI require corrent context-type header
    # But copilot client maybe not send such header
    request: Request,
    # copilot client ONLY request param
    engine: str,
):
    """_summary_
        From https://github.com/chenhunghan/ialacol/blob/main/main.py

        Similar to https://platform.openai.com/docs/api-reference/completions
        but with engine param and with /v1/engines
    Args:
        body (CompletionRequestBody): parsed request body
    Returns:
        StreamingResponse: streaming response
    """
    req_json = await request.json()
    log.debug("Body:%s", str(req_json))

    body = CompletionRequestBody(**req_json, model=engine)
    prompt = body.prompt
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = body.temperature if body.temperature else 0.85
    log.debug("temperature:%s", settings.temperature)
    settings.top_k = body.top_k if body.top_k else 50
    log.debug("top_k:%s", settings.top_k)
    settings.top_p = body.top_p if body.top_p else 0.8
    log.debug("top_p:%s", settings.top_p)
    settings.token_repetition_penalty = (
        body.repetition_penalty if body.repetition_penalty else 1.15
    )
    log.debug("token_repetition_penalty:%s", settings.token_repetition_penalty)
    tokenizer = app.state.tokenizer
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    max_new_tokens = body.max_tokens if body.max_tokens else 1024

    generator = request.app.state.generator
    generator.set_stop_conditions([tokenizer.eos_token_id])

    input_ids = tokenizer.encode(prompt)

    log.debug("Streaming response from %s", engine)

    def stream():
        generator.begin_stream(input_ids, settings)
        generated_tokens = 0
        while True:
            chunk, eos, _ = generator.stream()
            log.debug("Streaming chunk %s", chunk)
            created = times()
            generated_tokens += 1
            if eos or generated_tokens == max_new_tokens:
                stop_data = json.dumps(
                    {
                        "id": "id",
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": engine,
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
                yield f"data: {stop_data}" + "\n\n"
                break
            data = json.dumps(
                {
                    "id": "id",
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": engine,
                    "choices": [
                        {
                            "text": chunk,
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
            )
            yield f"data: {data}" + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
    )
