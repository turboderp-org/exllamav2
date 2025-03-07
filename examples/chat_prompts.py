
class PromptFormat:

    botname = "Chatbort"
    username = "User"

    def __init__(self):
        pass

    #

    def default_system_prompt(self):
        raise NotImplementedError

    def first_prompt(self, sysprompt: bool):
        raise NotImplementedError

    def subs_prompt(self):
        raise NotImplementedError

    def stop_conditions(self, tokenizer):
        raise NotImplementedError

    def encoding_options(self):  # (add_bos, add_eos, encode_special_tokens)
        raise NotImplementedError

    def print_bot_name(self):
        return False

    def print_extra_newline(self):
        return False


class PromptFormat_raw(PromptFormat):

    description = "Model-agnostic mode simulating a raw chatlog"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""This is a conversation between a helpful AI assistant named {self.botname} and a """ + \
            (f"""user named {self.username}.""" if self.username != "User" else """user.""")

    def first_prompt(self, sysprompt):
        if sysprompt:
            return f"""<|system_prompt|>\n{self.username}: <|user_prompt|>\n{self.botname}:"""
        else:
            return f"""{self.username}: <|user_prompt|>\n{self.botname}:"""

    def subs_prompt(self):
        return \
            f"""{self.username}: <|user_prompt|>\n{self.botname}:"""

    def stop_conditions(self, tokenizer):
        return \
            [self.username + ":",
             self.username[0:1] + ":",
             self.username.upper() + ":",
             self.username.lower() + ":",
             tokenizer.eos_token_id]

    def encoding_options(self):
        return True, False, False

    def print_bot_name(self):
        return True


class PromptFormat_llama(PromptFormat):

    description = "Llama-chat, Llama2-chat and Mistral-instruct models"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  """ + \
            """Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. """ + \
            """Please ensure that your responses are socially unbiased and positive in nature."""

    def first_prompt(self, sysprompt):
        if sysprompt:
            return """[INST] <<SYS>>\n<|system_prompt|>\n<</SYS>>\n\n<|user_prompt|> [/INST]"""
        else:
            return """[INST] <|user_prompt|> [/INST]"""

    def subs_prompt(self):
        return \
            """[INST] <|user_prompt|> [/INST]"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id]

    def encoding_options(self):
        return True, False, False

    def print_extra_newline(self):
        return True


class PromptFormat_llama3(PromptFormat):

    description = "Llama3-instruct models"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            """Assist users with tasks and answer questions to the best of your knowledge. Provide helpful and informative """ + \
            """responses. Be conversational and engaging. If you are unsure or lack knowledge on a topic, admit it and try """ + \
            """to find the answer or suggest where to find it. Keep responses concise and relevant. Follow ethical """ + \
            """guidelines and promote a safe and respectful interaction."""

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += \
                """<|start_header_id|>system<|end_header_id|>\n\n""" + \
                """<|system_prompt|><|eot_id|>"""
        r += \
            """<|start_header_id|>user<|end_header_id|>\n\n""" + \
            """<|user_prompt|><|eot_id|>""" + \
            """<|start_header_id|>assistant<|end_header_id|>\n\n"""
        return r

    def subs_prompt(self):
        return \
            """<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n""" + \
            """<|user_prompt|><|eot_id|>""" + \
            """<|start_header_id|>assistant<|end_header_id|>\n\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             tokenizer.single_id("<|eot_id|>"),
             tokenizer.single_id("<|start_header_id|>")]

    def encoding_options(self):
        return True, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_phi3(PromptFormat):

    description = "Phi3-instruct models"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            """You are a helpful AI assistant."""

    def first_prompt(self, sysprompt):
        r = """<s>"""
        if sysprompt:
            r += \
                """<|system|>\n""" + \
                """<|system_prompt|>""" + \
                """<|end|>\n"""
        r += \
            """<|user|>\n""" + \
            """<|user_prompt|><|end|>\n""" + \
            """<|assistant|>\n"""
        return r

    def subs_prompt(self):
        return \
            """<|end|>\n""" + \
            """<|user|>\n""" + \
            """<|user_prompt|>""" + \
            """<|end|>\n""" + \
            """<|assistant|>\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             tokenizer.single_id("<|end|>"),
             tokenizer.single_id("<|assistant|>"),
             tokenizer.single_id("<|endoftext|>")]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_codellama(PromptFormat_llama):

    description = "CodeLlama-instruct"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            """You are a helpful coding assistant. Always answer as helpfully as possible."""


class PromptFormat_qwq(PromptFormat):

    description = "Qwen QwQ format"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are a useful coding assistant, who thinks before answering."""

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += \
                """<|im_start|>system\n""" + \
                """<|system_prompt|>""" + \
                """<|im_end|>\n"""
        r += \
            """<|im_start|>user\n""" + \
            """<|user_prompt|><|im_end|>\n""" + \
            """<|im_start|>assistant\n<think>\n"""
        return r

    def subs_prompt(self):
        return \
            """<|im_end|>\n""" + \
            """<|im_start|>user\n""" + \
            """<|user_prompt|><|im_end|>\n""" + \
            """<|im_start|>assistant\n<think>\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             tokenizer.single_id("<|im_end|>"),
             """<|im_end|>"""]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_chatml(PromptFormat):

    description = "ChatML format, as used by e.g. (Mistral)Orca"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are {self.botname}, a large language model. Answer as concisely as possible."""

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += \
                """<|im_start|>system\n""" + \
                """<|system_prompt|>\n""" + \
                """<|im_end|>\n"""
        r += \
            """<|im_start|>user\n""" + \
            """<|user_prompt|><|im_end|>\n""" + \
            """<|im_start|>assistant\n"""
        return r

    def subs_prompt(self):
        return \
            """<|im_end|>\n""" + \
            """<|im_start|>user\n""" + \
            """<|user_prompt|><|im_end|>\n""" + \
            """<|im_start|>assistant\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             tokenizer.single_id("<|im_end|>"),
             """<|im_end|>"""]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_tinyllama(PromptFormat_chatml):

    description = "ChatML format, but ignoring special/added tokens. Use for TinyLlama-chat v0.3"

    def encoding_options(self):
        return False, False, False


class PromptFormat_zephyr(PromptFormat):

    description = "Zephyr 7b alpha prompt format."

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are {self.botname}, a large language model. Answer as concisely as possible."""

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += \
                """<|system|>\n""" + \
                """<|system_prompt|>\n""" + \
                """</s>\n"""
        r += \
            """<|user|>\n""" + \
            """<|user_prompt|></s>\n""" + \
            """<|assistant|>\n"""
        return r

    def subs_prompt(self):
        return \
            """<|user|>\n""" + \
            """<|user_prompt|></s>\n""" + \
            """<|assistant|>\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """<|user|>""",
             """</s>"""]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_deepseek(PromptFormat):

    description = "DeepSeek Coder Instruct"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."""

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += """<|system_prompt|>\n"""
        r += \
            """### Instruction:\n""" + \
            """<|user_prompt|>\n""" + \
            """### Response:\n"""
        return r

    def subs_prompt(self):
        return \
            """### Instruction:\n""" + \
            """<|user_prompt|>\n""" + \
            """### Response:\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """### Instruction"""]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_solar(PromptFormat):
    description = "Solar-instruct"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are an AI assistant."""

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += \
                """### System\n""" + \
                """<|system_prompt|>\n\n"""
        r += \
            """### User:\n""" + \
            """<|user_prompt|>\n\n""" + \
            """### Assistant:\n"""
        return r

    def subs_prompt(self):
        return \
            """### User:\n""" + \
            """<|user_prompt|>\n\n""" + \
            """### Assistant:\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """\n\n### User""",
             """\n### User""",
             ]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_openchat(PromptFormat):
    description = "openchat"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are an AI assistant."""

    def first_prompt(self, sysprompt):
        if sysprompt:
            return \
                """<|system_prompt|><|end_of_turn|>GPT4 Correct User:<|user_prompt|><|end_of_turn|>GPT4 Correct Assistant:"""
        else:
            return \
                """GPT4 Correct User:<|user_prompt|><|end_of_turn|>GPT4 Correct Assistant:"""

    def subs_prompt(self):
        return \
            """GPT4 Correct User:<|user_prompt|><|end_of_turn|>GPT4 Correct Assistant:"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """<|end_of_turn|>""",
             """<|endoftext|>""",
             """GPT4 Correct User:"""
             ]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_nous(PromptFormat):
    description = "Nous Research"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""Perform the task to the best of your ability."""

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += """<|system_prompt|>\n\n"""
        r += \
            """USER:\n""" + \
            """<|user_prompt|>\n\n""" + \
            """ASSISTANT:\n"""
        return r

    def subs_prompt(self):
        return \
            """USER:\n""" + \
            """<|user_prompt|>\n\n""" + \
            """ASSISTANT:\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """</s>""",
             ]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_gemma(PromptFormat):
    description = "Gemma"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return ""

    def first_prompt(self, sysprompt):
        return \
            """<bos><start_of_turn>user\n""" + \
            """<|user_prompt|><end_of_turn>\n""" + \
            """<start_of_turn>model\n"""

    def subs_prompt(self):
        return \
            """<end_of_turn>\n""" + \
            """<bos><start_of_turn>user\n""" + \
            """<|user_prompt|><end_of_turn>\n""" + \
            """<start_of_turn>model\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """</s>""",
             """<end_of_turn>""",
             ]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_granite(PromptFormat):
    description = "Granite (code)"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return "You are an AI coding assistant."

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += \
                """System:\n""" + \
                """<|system_prompt|>\n\n"""
        r += \
            """Question:\n""" + \
            """<|user_prompt|>\n\n""" + \
            """Answer:\n"""
        return r

    def subs_prompt(self):
        return \
            """\n\n""" + \
            """Question:\n""" + \
            """<|user_prompt|>\n\n""" + \
            """Answer:\n"""

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
            """\n\nQuestion:""",
        ]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_granite3(PromptFormat):
    description = "Granite 3"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return "You are Granite, developed by IBM. You are a helpful AI assistant."

    def first_prompt(self, sysprompt):
        r = ""
        if sysprompt:
            r += """<|start_of_role|>system<|end_of_role|>|system_prompt|><|end_of_text|>"""
        r += """<|start_of_role|>user<|end_of_role|><|user_prompt|><|end_of_text|>"""
        r += """<|start_of_role|>assistant<|end_of_role|>"""
        return r

    def subs_prompt(self):
        r = ""
        r += """<|start_of_role|>user<|end_of_role|><|user_prompt|><|end_of_text|>"""
        r += """<|start_of_role|>assistant<|end_of_role|>"""
        return r

    def stop_conditions(self, tokenizer):
        return [
            tokenizer.eos_token_id,
        ]

    def encoding_options(self):
        return True, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_cohere(PromptFormat):
    description = "Cohere"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are a helpful AI assistant."""

    def first_prompt(self, sysprompt):
        r = """<BOS_TOKEN>"""
        if sysprompt:
            r += \
                """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>""" + \
                """<|system_prompt|>""" + \
                """<|END_OF_TURN_TOKEN|>"""
        r += \
            """<|START_OF_TURN_TOKEN|><|USER_TOKEN|>""" + \
            """<|user_prompt|>""" + \
            """<|END_OF_TURN_TOKEN|>""" + \
            """<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""
        return r

    def subs_prompt(self):
        return \
            """<|END_OF_TURN_TOKEN|>""" + \
            """<|START_OF_TURN_TOKEN|><|USER_TOKEN|>""" + \
            """<|user_prompt|>""" + \
            """<|END_OF_TURN_TOKEN|>""" + \
            """<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """<|END_OF_TURN_TOKEN|>""",
             ]

    def encoding_options(self):
        return True, False, True

    def print_extra_newline(self):
        return True


prompt_formats = \
{
    "raw": PromptFormat_raw,
    "llama": PromptFormat_llama,
    "llama3": PromptFormat_llama3,
    "codellama": PromptFormat_codellama,
    "qwq": PromptFormat_qwq,
    "chatml": PromptFormat_chatml,
    "tinyllama": PromptFormat_tinyllama,
    "zephyr": PromptFormat_zephyr,
    "deepseek": PromptFormat_deepseek,
    "solar": PromptFormat_solar,
    "openchat": PromptFormat_openchat,
    "nous": PromptFormat_nous,
    "gemma": PromptFormat_gemma,
    "cohere": PromptFormat_cohere,
    "phi3": PromptFormat_phi3,
    "granite": PromptFormat_granite,
    "granite3": PromptFormat_granite3,
}
