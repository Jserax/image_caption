import os

import pandas as pd
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CaptionDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        max_len: int = 128,
        transform=None,
    ) -> None:
        super().__init__()
        data = pd.read_csv(annotations_file, sep="|")
        self.img_labels = data.image_name.tolist()
        self.img_dir = img_dir
        self.transform = transform
        tokenizer = AutoTokenizer.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D"
        )
        self.tokens = tokenizer.batch_encode_plus(
            data.comment.tolist(),
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple[FloatTensor, LongTensor]:
        image = Image.open(os.path.join(self.img_dir, self.img_labels[idx]))
        if self.transform:
            image = self.transform(image)
        tokens = self.tokens[idx]
        return image, tokens
