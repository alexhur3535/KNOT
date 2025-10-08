# src/models/Claude.py
from anthropic import Anthropic
from src.models.Model import Model

class Claude(Model):
    def __init__(self, config):
        super().__init__(config)
        api_pos = int(config["api_key_info"]["api_key_use"])
        api_key = config["api_key_info"]["api_keys"][api_pos]
        self.client = Anthropic(api_key=api_key)
        self.model_name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

    def query(self, msg: str) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
                system="You are a helpful assistant.",
                messages=[{"role":"user","content": msg}]
            )
            
            parts = getattr(resp, "content", [])
            return "".join([p.text for p in parts if getattr(p, "type", "") == "text"])
        except Exception as e:
            print("[Claude Error]", e)
            return ""


