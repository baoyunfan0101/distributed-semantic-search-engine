# vectorizer/bert_embedder.py

from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class BertEmbedder:

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            device: str = None,
            max_length: int = 512,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            outputs = self.model(**encoded)
            last_hidden = outputs.last_hidden_state  # (B, T, H)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)

            # Mean pooling
            masked_hidden = last_hidden * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            mean_pooled = summed / counts

            all_embeddings.append(mean_pooled.cpu().numpy())

        return np.vstack(all_embeddings)
