import os
from transformers import AutoTokenizer, BertModel, RobertaModel

def get_tokenlizer(text_encoder_type):
    print(f"final text_encoder_type: {text_encoder_type}")

    if text_encoder_type == "bert-base-uncased":
        local_bert_path = os.getenv("BERT_BASE_UNCASED_PATH")
        if local_bert_path and os.path.isdir(local_bert_path):
            print(f"从本地路径加载分词器: {local_bert_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_bert_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    else:
         print(f"为 {text_encoder_type} 从 Hugging Face Hub 加载分词器...")
         tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)

    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    print(f"get_pretrained_language_model received: {text_encoder_type}")

    if text_encoder_type == "bert-base-uncased":
        local_bert_path = os.getenv("BERT_BASE_UNCASED_PATH")
        if local_bert_path and os.path.isdir(local_bert_path):
            print(f"Identifier 'bert-base-uncased' detected. Using local path: {local_bert_path}")
            return BertModel.from_pretrained(local_bert_path)
        return BertModel.from_pretrained(text_encoder_type)

    elif os.path.isdir(text_encoder_type):
        print(f"Input is a directory path. Assuming BERT model and loading from: {text_encoder_type}")
        return BertModel.from_pretrained(text_encoder_type)

    elif text_encoder_type == "roberta-base":
        print(f"Loading RobertaModel for identifier: {text_encoder_type}")
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError("Unknown or unhandled text_encoder_type/path: {}".format(text_encoder_type))
