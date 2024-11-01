from typing import Union

import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import assert_tokenizer_consistency
from .metrics import perplexity, tv_perplexity, cos_perplexity, js_perplexity, entropy, mse, kl_divergence, total_variation_distance, cosine_similarity, js_divergence, focal_loss

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
BINOCULARS_ACCURACY_THRESHOLD = 1.0309

DEVICE_1 = "cuda:2" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "rinna/japanese-gpt-neox-3.6b",
                # observer_name_or_path: str = "rinna/japanese-gpt-neox-3.6b-instruction-sft",
                # observer_name_or_path: str = "cyberagent/calm2-7b",
                # observer_name_or_path: str = "elyza/ELYZA-japanese-Llama-2-7b",
                # observer_name_or_path: str = "tokyotech-llm/Swallow-7b-hf",
                 performer_name_or_path: str = "rinna/japanese-gpt-neox-3.6b-instruction-sft",
                # performer_name_or_path: str = "rinna/japanese-gpt-neox-3.6b",
                # performer_name_or_path: str = "cyberagent/calm2-7b-chat",
                # performer_name_or_path: str = "elyza/ELYZA-japanese-Llama-2-7b-instruct",
                # performer_name_or_path: str = "tokyotech-llm/Swallow-7b-instruct-hf",

                # observer_name_or_path: str = "tiiuae/falcon-7b",
                # performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

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
        self.observer_model.eval()
        self.performer_model.eval()

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

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        #ppl = tv_perplexity(encodings, performer_logits)
        #ppl = cos_perplexity(encodings, performer_logits)
        #ppl = 10 * js_perplexity(encodings, performer_logits)
        #x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
        #                encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        #x_ppl_sym = entropy(performer_logits.to(DEVICE_1), observer_logits.to(DEVICE_1),
        #                encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        #x_ppl = (x_ppl + x_ppl_sym)

        #x_ppl = mse(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1), encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        #x_ppl = 10000 * kl_divergence(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1), encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        #x_ppl = 10 * total_variation_distance(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1), encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        #x_ppl = 10 * cosine_similarity(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1), encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        #x_ppl = 100 * js_divergence(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1), encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        x_ppl = focal_loss(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1), encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()

        #binoculars_scores = ppl.tolist() #PPL単体の検証用
        #binoculars_scores = x_ppl.tolist() #X-PPL単体の検証用

        #print(binoculars_scores) #デバッグ用
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores # input_text がリスト（例: ["Text 1", "Text 2"]）であれば、compute_score はそれぞれのテキストに対するスコアを計算する

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
