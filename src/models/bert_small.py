"""Small BERT from scratch on encoded flows (plan §6.2).

Usage (from the repo root; run src.encoders.build and src.tokenizers.train_bpe first):
    uv run python -m src.models.bert_small --config configs/bert_small.yaml --stage tune    # LR grid, val only
    uv run python -m src.models.bert_small --config configs/bert_small.yaml --stage final   # best LR x seeds, test
    --stage all runs both. Optional filters: --encoding --dataset --task. Finished work is skipped unless --force.
"""

from transformers import BertConfig, BertForSequenceClassification

from src.config import load_bert_config
from src.models import text_runs
from src.models.text_train import tokenize
from src.tokenizers.train_bpe import load_tokenizer


def build_model(cfg, prep):
    config = BertConfig(vocab_size=prep["tok"].get_vocab_size(), max_position_embeddings=prep["meta"]["max_len"],
                        num_labels=len(prep["names"]), pad_token_id=prep["pad_id"], **cfg["model"])
    return BertForSequenceClassification(config)


def prepare(cfg, encoding, dataset, task, splits):
    """BPE tokenizer of the training dataset + tokenized train/val. Never touches the test split."""
    tok, meta = load_tokenizer(cfg, encoding, dataset)
    names, benign, texts = text_runs.train_val_texts(cfg, encoding, dataset, task, splits)
    data = {s: (tokenize(tok, t), y) for s, (t, y) in texts.items()}
    pad_id = tok.token_to_id(cfg["tokenizer"]["special_tokens"]["pad"])
    return {"tok": tok, "meta": meta, "names": names, "benign": benign, "pad_id": pad_id, "data": data}


BACKEND = text_runs.Backend(prepare=prepare, build_model=build_model)


def main():
    text_runs.main(BACKEND, load_bert_config, __doc__)


if __name__ == "__main__":
    main()
