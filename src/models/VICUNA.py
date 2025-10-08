import os, torch
from fastchat.model import load_model, get_conversation_template
from src.models.Model import Model

class Vicuna(Model):
    def __init__(self, config):
        super().__init__(config)

        self.name  = config["model_info"]["name"]     # HF 모델 id
        params     = config["params"]
        self.max_output_tokens  = int(params["max_output_tokens"])
        self.repetition_penalty = float(params.get("repetition_penalty", 1.0))
        self.load_8bit          = _to_bool(params.get("load_8bit", "false"))
        self.cpu_offloading     = _to_bool(params.get("cpu_offloading", "false"))
        self.max_gpu_memory     = params.get("max_gpu_memory", None)   # 예: "20GiB"
        self.revision           = params.get("revision", None)
        self.debug              = _to_bool(params.get("debug", "false"))


        gpus = params.get("gpus", None)  # ["0"], ["0","1"] or "0,1"
        if isinstance(gpus, str):
            gpus = [x.strip() for x in gpus.split(",") if x.strip()!=""]
        if not gpus: 
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            if vis and vis.strip() != "":
                gpus = [x.strip() for x in vis.split(",") if x.strip()!=""]
            else:
                if torch.cuda.is_available() and torch.cuda.device_count()>0:
                    gpus = ["0"]
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

        self.num_gpus = len(gpus) if gpus else 0
        device = "cuda" if (torch.cuda.is_available() and self.num_gpus>0) else "cpu"

        # --------- Load Model ----------
        self.model, self.tokenizer = load_model(
            model_path=self.name,
            device=device,
            num_gpus=self.num_gpus,
            max_gpu_memory=self.max_gpu_memory,
            load_8bit=self.load_8bit,
            cpu_offloading=self.cpu_offloading,
            revision=self.revision,
            debug=self.debug
        )

    def query(self, msg: str) -> str:
        try:
            conv = get_conversation_template(self.name) or get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], msg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = self.tokenizer([prompt]).input_ids
            import torch
            input_ids = torch.tensor(inputs).to(self.model.device)

            out_ids = self.model.generate(
                input_ids=input_ids,
                do_sample=(self.temperature>0),
                temperature=float(self.temperature),
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_output_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
            )
            gen = out_ids[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        except Exception as e:
            print("[Vicuna Error]", e)
            return ""

def _to_bool(v):
    return (str(v).lower() == "true")
