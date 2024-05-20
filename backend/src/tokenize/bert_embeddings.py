from tqdm.auto import tqdm
import torch
import torch.utils.data


class PrepareData:
    """
    Представление текстов с помощью bert-эмбеддингов
    """

    def __init__(self, texts, tokenizer, batch_size_split=10, max_length=512):

        self.texts = texts
        self.tokenizer = tokenizer
        self.batch_size_split = batch_size_split
        self.max_length = max_length

    def pre_tokenizer(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def transform(self):
        N = len(self.texts)
        size_split = N // self.batch_size_split

        train_encodings = self.pre_tokenizer(self.texts[:size_split])
        input_ids = train_encodings["input_ids"]
        attention_mask = train_encodings["attention_mask"]
        token_type_ids = train_encodings["token_type_ids"]

        for pos in tqdm(range(size_split, N, size_split)):
            train_encodings_2 = self.pre_tokenizer(self.texts[pos : pos + size_split])
            input_ids = torch.cat((input_ids, train_encodings_2["input_ids"]))
            attention_mask = torch.cat(
                (attention_mask, train_encodings_2["attention_mask"])
            )
            token_type_ids = torch.cat(
                (token_type_ids, train_encodings_2["token_type_ids"])
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


class CustomDataset(torch.utils.data.Dataset):
    """
    Класс представления датасета, содержащего объекты-эмбеддинги, поддерживающий
    итерацию для использования с объектом Trainer
    """

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Если датасет содержит метки класса
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
