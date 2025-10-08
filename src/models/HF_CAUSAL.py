import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from src.models.Model import Model

class HFCausal(Model):
    """
    HuggingFace Causal LM Wrapper
    - model_info.name: HF model id (예: "mistralai/Mistral-7B-Instruct-v0.3")
    - api_key_info.api_keys[api_key_use]: HF Token
    - params: device, max_output_tokens, temperature, quantization("4bit"|None), use_chat_template(bool)
    """
    def __init__(self, config):
        super().__init__(config)
        self.model_id = config["model_info"]["name"]
        self.max_new_tokens = int(config["params"]["max_output_tokens"])
        self.temperature = float(config["params"].get("temperature", 0.2))
        self.use_chat_template = bool(config["params"].get("use_chat_template", True))

        # Loggin Info
        api = config.get("api_key_info", {})
        idx = int(api.get("api_key_use", 0))
        keys = api.get("api_keys", [])
        if keys and 0 <= idx < len(keys) and keys[idx]:
            login(token=keys[idx])

        # Loading Option
        quant = config["params"].get("quantization")
        load_kwargs = {"device_map": "auto"}
        if quant == "4bit":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

    def _build_inputs(self, msg: str):
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        return self.tokenizer(msg, return_tensors="pt").to(self.model.device)

    def query(self, msg: str) -> str:
        inputs = self._build_inputs(msg)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=(self.temperature > 0),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        gen_ids = gen[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
