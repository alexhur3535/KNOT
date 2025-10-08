# src/models/Gemini.py
import google.generativeai as genai
from src.models.Model import Model

class Gemini(Model):
    def __init__(self, config):
        super().__init__(config)
        api_pos = int(config["api_key_info"]["api_key_use"])
        api_key = config["api_key_info"]["api_keys"][api_pos]
        genai.configure(api_key=api_key)
        self.model_name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        
        self.model = genai.GenerativeModel(self.model_name)

    def query(self, msg: str) -> str:
        try:
            resp = self.model.generate_content(
                msg,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            return resp.text or ""
        except Exception as e:
            print("[Gemini Error]", e)
            return ""
