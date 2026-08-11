import os

import torch


UNIGRAM1000_LIST = (
    ['<blank>']
    + [_.split()[0] for _ in open(os.path.join(os.path.dirname(__file__), "labels", "unigram1000_units.txt"), encoding="utf-8").read().splitlines()]
    + ['<eos>']
)


_UNIGRAM_TOKEN_TO_ID = None
_UNIGRAM_ID_TO_TOKEN = None
_UNIGRAM_MAX_TOKEN_LEN = None


def load_unigram_map():
    global _UNIGRAM_TOKEN_TO_ID, _UNIGRAM_ID_TO_TOKEN, _UNIGRAM_MAX_TOKEN_LEN
    if _UNIGRAM_TOKEN_TO_ID is not None and _UNIGRAM_ID_TO_TOKEN is not None:
        return _UNIGRAM_TOKEN_TO_ID, _UNIGRAM_ID_TO_TOKEN

    label_path = os.path.join(os.path.dirname(__file__), "labels", "unigram1000_units.txt")
    token_to_id = {}
    id_to_token = {}

    with open(label_path, "r", encoding="utf-8") as file:
        for raw in file:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            token, idx = raw.rsplit(" ", 1)
            token_id = int(idx)
            token_to_id[token] = token_id
            id_to_token[token_id] = token

    _UNIGRAM_TOKEN_TO_ID = token_to_id
    _UNIGRAM_ID_TO_TOKEN = id_to_token
    _UNIGRAM_MAX_TOKEN_LEN = max(len(token) for token in token_to_id)
    return _UNIGRAM_TOKEN_TO_ID, _UNIGRAM_ID_TO_TOKEN


def str_to_ids(text: str):
    token_to_id, _ = load_unigram_map()
    unk_id = token_to_id.get("<unk>", 1)
    max_token_len = _UNIGRAM_MAX_TOKEN_LEN or max(len(token) for token in token_to_id)

    normalized = text.replace(" ", "▁")
    token_ids = []
    position = 0
    text_len = len(normalized)

    while position < text_len:
        matched_piece = None
        search_end = min(text_len, position + max_token_len)
        for end in range(search_end, position, -1):
            piece = normalized[position:end]
            if piece in token_to_id:
                matched_piece = piece
                break

        if matched_piece is not None:
            token_ids.append(token_to_id[matched_piece])
            position += len(matched_piece)
        else:
            token_ids.append(token_to_id.get(normalized[position], unk_id))
            position += 1

    return token_ids


def ids_to_str_from_ids(token_ids):
    _, id_to_token = load_unigram_map()
    tokens = [id_to_token.get(int(token_id), "<unk>") for token_id in token_ids]
    return "".join(tokens).replace("▁", " ").strip()


def ids_to_str(token_ids, char_list):
    tokenid_as_list = list(map(int, token_ids))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    return "".join(token_as_list).replace("<space>", " ")


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
