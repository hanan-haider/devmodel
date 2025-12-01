""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import torch
import ftfy
import html
from typing import Callable, List, Optional, Union, Dict
import warnings


# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_nltk_init = False


DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP



def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()



def whitespace_clean(text):
    text = " ".join(text.split())
    text = text.strip()
    return text

def _clean_whitespace(x):
    # basic, remove whitespace
    return whitespace_clean(basic_clean(x))

def get_clean_fn(type: str):
    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."

      


class HFTokenizer:
    """HuggingFace tokenizer wrapper with support for custom tokenization modes"""

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'whitespace',
            strip_sep_token: bool = False,
            language: Optional[str] = None,
            cache_dir: Optional[str] = None,
            tokenizer_mode: Optional[str] = None,  # None, 'clips'
            **kwargs
    ):
        self.tokenizer_mode = tokenizer_mode or ''
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token

        # NOTE: Left as example of loading custom tokenizer from file for experimentation
        # if self.tokenizer_mode == 'bert_clips':
        #     self.special_tokens = {
        #         "bos_token": 1,
        #         "eos_token": 2,
        #         "cls_token": 101,
        #         "pad_token": 0
        #     }
        #
        #     # For BERT CLIPS mode with vocab file
        #     from tokenizers import BertWordPieceTokenizer
        #     if tokenizer_name.startswith('hf-hub:'):
        #         from huggingface_hub import hf_hub_download
        #         # Format: hf-hub:repo_id/filename
        #         repo_url = tokenizer_name[7:]
        #         parts = repo_url.split('/')
        #         filename = parts[-1]
        #         repo_id = '/'.join(parts[:-1])
        #         vocab_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        #         self.tokenizer = BertWordPieceTokenizer(lowercase=True)
        #         self.tokenizer = self.tokenizer.from_file(vocab_file)
        #     else:
        #         # Assume tokenizer_name is a local path to a vocab file
        #         self.tokenizer = BertWordPieceTokenizer(lowercase=True)
        #         self.tokenizer = self.tokenizer.from_file(tokenizer_name)

        # Standard HuggingFace tokenizer initialization
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            **kwargs
        )

        # Set language function if available
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self.clean_fn(text) for text in texts]

        # Handle different tokenization modes
        if self.tokenizer_mode == 'clips':
            return self._clips_tokenize(texts, context_length)
        else:
            # Standard tokenization
            input_ids = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors='pt',
                max_length=context_length,
                padding='max_length',
                truncation=True,
            ).input_ids

            if self.strip_sep_token:
                input_ids = torch.where(
                    input_ids == self.tokenizer.sep_token_id,
                    torch.zeros_like(input_ids),
                    input_ids,
                )

            return input_ids

    def set_language(self, src_lang):
        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn('Cannot set language for the tokenizer.')

    def _clips_tokenize(self, texts: List[str], context_length: int) -> torch.Tensor:
        """Use standard HF tokenizer but apply custom post-processing"""
        # Use standard tokenizer without special tokens - we'll add our own
        encoded_outputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )

        encoded = []
        for tokens in encoded_outputs["input_ids"]:
            tokens = tokens[:context_length - 3]  # Leave room for special tokens
            tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            encoded.append(tokens)

        # Create result tensor and handle padding + class token
        result = torch.zeros(len(encoded), context_length, dtype=torch.long)
        for i, tokens in enumerate(encoded):
            padded_tokens = self._pad_and_add_class_token(
                tokens,
                max_length=context_length,
                pad_token_id=self.tokenizer.pad_token_id,
                cls_token_id=self.tokenizer.cls_token_id,
            )
            result[i, :len(padded_tokens)] = torch.tensor(padded_tokens)

        return result

    def _pad_and_add_class_token(
            self,
            tokens: List[int],
            max_length: int,
            pad_token_id: int = 0,
            cls_token_id: int = 101,
    ) -> List[int]:
        """ Add padding with class token at the end """
        if len(tokens) > max_length - 1:
            tokens = tokens[:max_length - 1]

        # Add padding to reach max_length-1
        if len(tokens) < max_length - 1:
            tokens = tokens + [pad_token_id] * (max_length - 1 - len(tokens))

        # Add class token at the end
        tokens = tokens + [cls_token_id]
        return tokens

tokenize = HFTokenizer(
    tokenizer_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    context_length=256
)
