from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")


def tokenize_with_mistral(text: str) -> list[str]:
    return tokenizer.tokenize(text)
