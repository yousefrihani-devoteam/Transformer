import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


class TextDataPrep:
    def __init__(self, dataset_name="roneneldan/TinyStories", tokenizer_name="gpt2",
                 max_src=128, max_tgt=128):
        self.dataset_name = dataset_name
        self.max_src = max_src
        self.max_tgt = max_tgt

        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tok.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        self.bos_id = self.tok.bos_token_id
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id

    def load(self):
        ds = load_dataset(self.dataset_name)
        return ds.filter(lambda x: len(x["text"].strip()) > 0)

    def _make_pairs(self, batch):
        src, dec_in, labels = [], [], []
        for ids in self.tok(batch["text"])["input_ids"]:
            if len(ids) < 8:
                continue
            cut = len(ids) // 2
            s = ids[:cut][:self.max_src]
            t = ids[cut:][:self.max_tgt - 1]
            src.append(s)
            dec_in.append([self.bos_id] + t)
            labels.append(t + [self.eos_id])
        return {"src": src, "dec_in": dec_in, "labels": labels}

    def prepare(self):
        ds = self.load()
        return ds.map(self._make_pairs, batched=True, remove_columns=["text"])

    def collate(self, batch):
        def pad(key, value):
            return pad_sequence([torch.tensor(b[key]) for b in batch],
                                batch_first=True, padding_value=value)
        return {
            "src":    pad("src", self.pad_id),
            "dec_in": pad("dec_in", self.pad_id),
            "labels": pad("labels", -100),
        }