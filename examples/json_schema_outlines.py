# Install Outlines:
# pip install outlines

# Download Model:
# huggingface-cli download bartowski/Phi-3.1-mini-4k-instruct-exl2 --revision 6_5 --local-dir Phi-3.1-mini-4k-instruct-exl2-6_5

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler

from outlines.processors import JSONLogitsProcessor
from outlines.models.exllamav2 import patch_tokenizer as patch_exl2_tokenizer_for_outlines

from pydantic import BaseModel, Field, RootModel
from typing import Optional, Union, Literal
from datetime import time


################################################
# Create Structured JSON Generator With Outlines
################################################

# Additional Examples: https://outlines-dev.github.io/outlines/cookbook/
# JSON Generation Docs: https://outlines-dev.github.io/outlines/reference/json/
# `outlines.processors` also supports guaranteed regex patterns and lark grammars

# Example: Home Assistant extension for natural language commands -> actions
class LightAction(BaseModel):
    entity: Literal["light"] = "light"
    action: Literal["turn_on", "turn_off", "set_brightness"]
    brightness: Optional[int] = Field(None, ge=0, le=100)
    execute_at: Optional[time] = None


class OvenAction(BaseModel):
    entity: Literal["oven"] = "oven"
    action: Literal["turn_on", "turn_off", "set_temperature"]
    temperature: Optional[float] = Field(None, ge=50, le=300)
    execute_at: Optional[time] = None


class HomeAssistantAction(BaseModel):
    instruction: Union[LightAction, OvenAction]


def create_generator(model_dir="/mnt/str/models/mistral-7b-exl2/4.0bpw"):
    config = ExLlamaV2Config(model_dir)
    config.arch_compat_overrides()
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=32768, lazy=True)
    model.load_autosplit(cache, progress=True)

    print("Loading tokenizer...")
    tokenizer = ExLlamaV2Tokenizer(config)
    tokenizer.vocabulary = tokenizer.extended_piece_to_id

    # Initialize the generator with all default parameters
    return ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
    )


generator = create_generator("./Phi-3.1-mini-4k-instruct-exl2-6_5")

gen_settings = ExLlamaV2Sampler.Settings()
gen_settings.logits_processor = JSONLogitsProcessor(
    HomeAssistantAction,
    patch_exl2_tokenizer_for_outlines(generator.tokenizer)
)


rules = "JSON for an instruction with an entity (light or oven) and action (turn_on, turn_off, set_brightness, set temperature). *Optionally* you may set an execute_at time-of-day if the user specifies, otherwise set to null"
prompts = [
    f"<|user|> {rules} Turn the lights lower please!<|end|><|assistant|>",
    f"<|user|> {rules} I need the oven set for homemade pizza when I get home from work at 6PM.<|end|><|assistant|>",
    f"<|user|> {rules} Oh no the lights are off and I can't find the switch!<|end|><|assistant|>",
]

outputs = generator.generate(
    prompt=prompts,
    gen_settings=gen_settings,
    max_new_tokens=2048,
    completion_only=True,
    encode_special_tokens=False,
    stop_conditions=[generator.tokenizer.eos_token_id],
)

# raw json format
for idx, output in enumerate(outputs):
    print(output)
# Output:
# {"instruction": {"entity": "light", "action": "set_brightness", "execute_at": null}}
# {"instruction": {"entity": "oven", "action": "set_temperature", "execute_at": "18:00:00"} }
# {"instruction": {"entity": "light", "action": "turn_on"}}

# pydantic model format
for idx, output in enumerate(outputs):
    print(repr(HomeAssistantAction.parse_raw(output)))
# Output:
# HomeAssistantAction(instruction=LightAction(entity='light', action='set_brightness', brightness=None, execute_at=None))
# HomeAssistantAction(instruction=OvenAction(entity='oven', action='set_temperature', temperature=None, execute_at=datetime.time(18, 0)))
# HomeAssistantAction(instruction=LightAction(entity='light', action='turn_on', brightness=None, execute_at=None))
