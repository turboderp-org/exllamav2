
# from exllamav2 import (
#     ExLlamaV2,
#     ExLlamaV2Config,
#     ExLlamaV2Cache,
#     ExLlamaV2Tokenizer
# )

import json

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

async def dispatch(request, ws, server):

    request_id = request["request_id"]
    action_ = request["action"]
    response = { "action": action_, "request_id": request_id }

    if action_ == "echo": echo(request, ws, server, response)
    elif action_ == "estimate_token": estimate_token(request, ws, server, response)
    elif action_ == "lefttrim_token": lefttrim_token(request, ws, server, response)
    try:
    	else: await infer(request, ws, server, response)
    except:
    	print(f" ## Unknown request from client: {request}")
        return

    await ws.send(json.dumps(response))


def echo(request, ws, server, response):

    """
    request:  { action: str = "echo",
                request_id: str }

    response: { action: str = "echo",
                request_id: str }
    """

    pass


def estimate_token(request, we, server, response):

    """
    request:  { action: str = "estimate_token",
                request_id: str,
                text: str }                         # text to measure

    response: { action: str = "estimate_token",
                request_id: str,
                num_tokens: int }                   # length of input text, in tokens
    """

    text = request["text"]
    ids = server.tokenizer.cached_encode_str(text)
    response["num_tokens"] = ids.shape[-1]


def lefttrim_token(request, ws, server, response):

    """
    request:  { action: str = "lefttrim_token",
                request_id: str,
                text: str,                          # text to trim
                trimmed_length: int }               # num tokens to keep, from right

    response: { action: str = "lefttrim_token",
                request_id: str,
                trimmed_text: str }                 # input, trimmed
    """

    text = request["text"]
    length = int(request["trimmed_length"])

    ids = server.tokenizer.cached_encode_str(text)
    if ids.shape[-1] <= length:
        response["trimmed_text"] = text
    else:
        response["trimmed_text"] = server.tokenizer.decode(ids[:, -length:])[0]


async def infer(request, ws, server, response):

    """
    request:  { action: str = "infer",
                request_id: str,
                text: str,                          # input prompt
                max_new_tokens: int,                # max num new tokens
                stream: bool,                       # stream response
                top_p: float,                       # (optional) top-P threshold (0 to disable)
                top_k: int,                         # (optional) top-K count (0 to disable)
                typical: float,                     # (optional) typical threshold (0 to disable)
                temperature: float,                 # (optional) sampling temperature (1.0 = no temp adjust)
                rep_pen: float,                     # (optional) repetition penalty (1.0 = no penalty)
                stop_conditions: [str|int],         # (optional) list of stop conditions
                token_healing: bool }               # (optionsl) enable token healing

    streams:  { action: str = "infer",
                request_id: str,
                response_type: str = "chunk",
                chunk: str }                        # next chunk of response

    response: { action: str = "infer",
                request_id: str,
                response_type: str = "full",
                util_text: str,                     # input context (pruned if max_seq_len exceeded)
                response: str }                     # full response excluding input prompt
    """

    # Mode

    stream = request["stream"]

    # Stop conditions

    sc = [server.tokenizer.eos_token_id]
    if "stop_conditions" in request:
        ss = request["stop_conditions"].split(',')
        sc += ss

    # Tokenize and trim prompt

    full_ctx = request["text"]
    num_tokens = request["max_new_tokens"]

    ids = server.tokenizer.cached_encode_str(full_ctx)
    overflow = ids.shape[-1] + num_tokens - server.model.config.max_seq_len
    if overflow > 0:
        ids = ids[:, overflow:]
        util_ctx = server.tokenizer.decode(ids)
    else:
        util_ctx = full_ctx

    # Sampler

    gs = ExLlamaV2Sampler.Settings()
    gs.top_k = int(request["top_k"]) if "top_k" in request else 100
    gs.top_p = float(request["top_p"]) if "top_p" in request else 0.8
    gs.typical = float(request["typical"]) if "typical" in request else 0
    gs.temperature = float(request["temperature"]) if "temperature" in request else 0.95
    gs.token_repetition_penalty = float(request["rep_pen"]) if "rep_pen" in request else 1.15

    # Generate

    server.generator.set_stop_conditions(sc)
    server.generator.begin_stream(ids, gs, token_healing = request["token_healing"] if "token_healing" in request else False)

    completion = ""
    gen_tokens = 0

    while True:
        chunk, eos, _ = server.generator.stream()
        completion += chunk
        gen_tokens += 1

        if stream and chunk != "":
            response["response_type"] = "chunk"
            response["chunk"] = chunk
            await ws.send(json.dumps(response))

        if eos or gen_tokens >= num_tokens: break

    if stream: del response["chunk"]
    response["response_type"] = "full"
    response["util_text"] = util_ctx
    response["response"] = completion
