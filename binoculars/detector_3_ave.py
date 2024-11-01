from typing import Union

import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
# BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
# BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]
# BINOCULARS_ACCURACY_THRESHOLD = 0.9011  # [rinna3.6b_livedoor] optimized for f1-score
# BINOCULARS_FPR_THRESHOLD = 0.856521725654602  # [rinna3.6b_livedoor] optimized for low-fpr [chosen at 0.01%]
# BINOCULARS_ACCURACY_THRESHOLD = 0.8819  # [rinna3.6b_oscar] optimized for f1-score
# BINOCULARS_FPR_THRESHOLD = 0.8318965435028076  # [rinna3.6b_oscar] optimized for low-fpr [chosen at 0.01%]
BINOCULARS_ACCURACY_THRESHOLD = 1.031

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1
DEVICE_3 = "cuda:2" if torch.cuda.device_count() > 2 else (DEVICE_2 if torch.cuda.device_count() > 1 else DEVICE_1) # **追加**



class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "rinna/japanese-gpt-neox-3.6b",
                # performer_name_or_path: str = "rinna/japanese-gpt-neox-3.6b",
                 performer_name_or_path: str = "rinna/japanese-gpt-neox-3.6b-instruction-sft",
                # observer_name_or_path: str = "tiiuae/falcon-7b",
                # performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 third_model_name_or_path: str = "rinna/japanese-gpt-neox-3.6b-instruction-ppo",  # **3つ目のモデル**
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path) # トークナイザの一致を確認
        assert_tokenizer_consistency(performer_name_or_path, third_model_name_or_path) # **トークナイザの一致を確認**

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"]
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"]
                                                                    )
        self.third_model = AutoModelForCausalLM.from_pretrained(third_model_name_or_path,  # **3つ目のモデルの初期化**
                                                                device_map={"": DEVICE_1},
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16 if use_bfloat16
                                                                else torch.float32,
                                                                token=huggingface_config["TOKEN"]
                                                                )
        self.observer_model.eval()
        self.performer_model.eval()
        self.third_model.eval()  # **3つ目のモデルも評価モードに設定**

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode() # それぞれについてlogitsを計算
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        third_logits = self.third_model(**encodings.to(DEVICE_1)).logits # **追加**
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits, third_logits # **追加**

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits, third_logits = self._get_logits(encodings) # **追加**
        """
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        """
        # **各スコアを計算**
        binoculars_score_1 = self.calculate_binoculars_score(encodings, observer_logits, performer_logits, DEVICE_2)
        binoculars_score_2 = self.calculate_binoculars_score(encodings, performer_logits, third_logits, DEVICE_2)
        binoculars_score_3 = self.calculate_binoculars_score(encodings, third_logits, observer_logits, DEVICE_2)

        # **3つのスコアの平均を計算**
        binoculars_scores = (binoculars_score_1 + binoculars_score_2 + binoculars_score_3) / 3
        binoculars_scores = binoculars_scores.tolist()

        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def calculate_binoculars_score(self, encodings, logits_1, logits_2, device):
        # encodings, logits_1, logits_2 すべてのテンソルを共通のデバイスに移動
        encodings = encodings.to(device)
        logits_1 = logits_1.to(device)
        logits_2 = logits_2.to(device)

        ppl = perplexity(encodings, logits_2)
        x_ppl = entropy(logits_1, logits_2, encodings, self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        return binoculars_scores #これはスカラー値

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
