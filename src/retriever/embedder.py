import os
import typing as tp

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class Embedder(nn.Module):

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        device: str,
    ):
        super().__init__()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def from_resources_path(resources_path: str, device: tp.Optional[str] = None) -> "Embedder":
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        model = AutoModel.from_pretrained(resources_path)
        tokenizer = AutoTokenizer.from_pretrained(resources_path)
        return Embedder(tokenizer, model, device)

    @staticmethod
    def _cls_pooling(model_output):
        return model_output[0][:, 0, :]
    
    def _tokenize(
        self,
        text: str
    ) -> tp.Dict[str, torch.Tensor]:
        
        return self._tokenizer(
            text,
            return_tensors='pt',
            truncation=True
        )
    
    @torch.no_grad()
    def __call__(
        self,
        text: str
    ) -> torch.Tensor:
        
        input_tokens = self._tokenize(text)
        model_output = self._model(**{k: v.to(self._device) for k, v in input_tokens.items()})
        cls_pooling = self._cls_pooling(model_output)
        
        resp = cls_pooling / torch.norm(cls_pooling, dim=-1, keepdim=True)
        return resp.cpu().numpy()[0]
