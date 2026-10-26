"""BERT tokenization used by the text branch."""
import numpy as np
from transformers import BertTokenizer


class TextTokenizer:
    def __init__(self, pretrained_path, max_len):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path, do_lower_case=True)
        self.max_len = int(max_len)

    def __call__(self, texts):
        rows = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_len - 2:
                tokens = tokens[:self.max_len - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            padding = [0] * (self.max_len - len(input_ids))
            rows.append([input_ids + padding,
                         input_mask + padding,
                         segment_ids + padding])
        return np.asarray(rows, dtype=np.int64)
