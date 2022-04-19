import json
import os

import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from preprocess.data_utils import infer_column_type_from_sampled_value
from table_bert import Table, Column

nlp_model = spacy.load('en_core_web_sm')


def encode_file_bart(tokenizer, texts, max_length, pad_to_max_length=True, return_tensors="pt"):
    examples = []
    for text in tqdm(texts):
        tokenized = tokenizer.batch_encode_plus(
            [text.strip()], max_length=max_length, pad_to_max_length=pad_to_max_length,
            return_tensors=return_tensors
        )
        examples.append(tokenized)
    return examples


def get_tables(data_dir='./data/dataset/train/few-shot/', file='train.json', tokenizer=None):
    path = os.path.join(data_dir, file)

    with open(path, 'r') as f:
        data_dict = json.load(f)

    tables, contexts = [], []

    for key, sample_dict in tqdm(data_dict.items()):
        first_row = sample_dict["table_content_values"][0]
        header = []
        for i, col_name in enumerate(sample_dict["table_column_names"]):
            sample_value_entry = {
                'value': first_row[i],
            }
            annotation = nlp_model(first_row[i])
            tokenized_value = [token.text for token in annotation]
            ner_tags = [token.ent_type_ for token in annotation]
            pos_tags = [token.pos_ for token in annotation]

            sample_value_entry.update({
                'tokens': tokenized_value,
                'ner_tags': ner_tags,
                'pos_tags': pos_tags
            })
            col_type = infer_column_type_from_sampled_value(sample_value_entry)

            header.append(Column(col_name, col_type, sample_value_entry['value']))

        table = Table(
            id=sample_dict["table_caption"],
            header=header,
            data=sample_dict["table_content_values"]
        )

        if tokenizer is not None:
            table = table.tokenize(tokenizer)
            context = tokenizer.tokenize(sample_dict["table_caption"])
        else:
            context = sample_dict["table_caption"]

        tables.append(table)
        contexts.append(context)

    return tables, contexts


class TableDataset(Dataset):
    def __init__(
            self,
            tokenizer_encoder,
            tokenizer_decoder,
            data_dir=None,
            type_path="train",
            max_target_length=512
    ):
        super().__init__()
        self.encoder_tokenizer = tokenizer_encoder
        self.decoder_tokenizer = tokenizer_decoder

        self.tables, self.contexts = get_tables(tokenizer=tokenizer_encoder)

        self.target = encode_file_bart(tokenizer_decoder, os.path.join(data_dir, type_path + ".target"),
                                       max_target_length)

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, index):
        target_ids = self.target[index]["input_ids"].squeeze()
        table = self.tables[index]
        context = self.contexts[index]
        return {"table": table, "context": context, "target_ids": target_ids}

    def collate_fn(self, batch):
        table = torch.stack([x["table"] for x in batch])
        context = torch.stack([x["context"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        return {"table": table, "context": context, "target_ids": target_ids}
