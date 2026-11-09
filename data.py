from datasets import load_dataset
from transformers import AutoTokenizer



class TextDataPrep:
    def __init__(self, dataset_name, config=None, tokenizer_name="gpt2",
                block_size=256, add_eos=True):
        self.dataset_name = dataset_name
        self.config = config
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.block_size = block_size
        self.add_eos = add_eos

    def load(self):
        ds = load_dataset(self.dataset_name, self.config)
        return ds.filter(lambda x: len(x["text"].strip()) > 0)

    def _tokenize(self, batch):
        texts = batch["text"]
        if self.add_eos:
            texts = [t + self.tok.eos_token for t in texts]
        return self.tok(texts)

    def _group(self, batch):
        ids = sum(batch["input_ids"], [])
        n = (len(ids) // self.block_size) * self.block_size
        chunks = [ids[i:i + self.block_size] for i in range(0, n, self.block_size)]
        return {"input_ids": chunks, "labels": chunks}

    def prepare(self):
        ds = self.load()
        ds = ds.map(self._tokenize, batched=True, remove_columns=["text"])
        ds = ds.map(self._group, batched=True,
                    remove_columns=ds["train"].column_names)
        return ds