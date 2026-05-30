from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Dict, List, Tuple


class CustomByteLevelBPETokenizer:
    """
    Custom Byte-Level BPE Tokenizer.

    Features:
    - Byte-level tokenization
    - BPE training
    - Encoding / Decoding
    - Save / Load model
    """

    def __init__(self, vocab_size: int = 1000):

        if vocab_size < 260:
            raise ValueError(
                "vocab_size must be >= 260 "
                "(256 bytes + 4 special tokens)"
            )

        self.vocab_size = vocab_size

        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        self.vocab: Dict[bytes, int] = {}

        # Initialize byte vocabulary
        for i in range(256):
            self.vocab[bytes([i])] = (
                i + len(self.special_tokens)
            )

        # Add special tokens
        for token, token_id in self.special_tokens.items():
            self.vocab[token.encode()] = token_id

        self.inverse_vocab: Dict[int, bytes] = {
            idx: token
            for token, idx in self.vocab.items()
        }

        self.merges: List[
            Tuple[bytes, bytes]
        ] = []

    def text_to_bytes(
        self,
        text: str
    ) -> List[int]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        return list(text.encode("utf-8"))

    def get_pair_frequencies(
        self,
        corpus: List[List[bytes]]
    ) -> Dict[Tuple[bytes, bytes], int]:

        pair_freq = defaultdict(int)

        for tokens in corpus:
            for i in range(len(tokens) - 1):
                pair = (
                    tokens[i],
                    tokens[i + 1]
                )
                pair_freq[pair] += 1

        return pair_freq

    def merge_pair(
        self,
        pair: Tuple[bytes, bytes],
        corpus: List[List[bytes]]
    ) -> List[List[bytes]]:

        updated_corpus = []

        for tokens in corpus:

            merged_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    merged_tokens.append(
                        tokens[i] + tokens[i + 1]
                    )
                    i += 2

                else:
                    merged_tokens.append(tokens[i])
                    i += 1

            updated_corpus.append(
                merged_tokens
            )

        return updated_corpus

    def train(
        self,
        corpus: List[str]
    ) -> None:

        if not corpus:
            raise ValueError(
                "Corpus cannot be empty"
            )

        tokenized_corpus = []

        for text in corpus:
            tokens = [
                bytes([b])
                for b in self.text_to_bytes(text)
            ]
            tokenized_corpus.append(tokens)

        num_merges = (
            self.vocab_size -
            len(self.vocab)
        )

        print("=" * 50)
        print("Training Byte-Level BPE")
        print("=" * 50)

        for step in range(
            max(0, num_merges)
        ):

            pair_freq = (
                self.get_pair_frequencies(
                    tokenized_corpus
                )
            )

            if not pair_freq:
                break

            best_pair = max(
                pair_freq,
                key=pair_freq.get
            )

            merged_token = (
                best_pair[0]
                + best_pair[1]
            )

            self.merges.append(
                best_pair
            )

            if (
                merged_token
                not in self.vocab
            ):
                token_id = len(
                    self.vocab
                )

                self.vocab[
                    merged_token
                ] = token_id

                self.inverse_vocab[
                    token_id
                ] = merged_token

            tokenized_corpus = (
                self.merge_pair(
                    best_pair,
                    tokenized_corpus
                )
            )

            if (
                step + 1
            ) % 100 == 0:
                print(
                    f"Completed "
                    f"{step + 1} merges"
                )

        print(
            f"Training Complete. "
            f"Vocabulary Size: "
            f"{len(self.vocab)}"
        )

    def apply_bpe(
        self,
        byte_tokens: List[int]
    ) -> List[bytes]:

        tokens = [
            bytes([b])
            for b in byte_tokens
        ]

        for pair in self.merges:

            merged_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    merged_tokens.append(
                        tokens[i]
                        + tokens[i + 1]
                    )
                    i += 2

                else:
                    merged_tokens.append(
                        tokens[i]
                    )
                    i += 1

            tokens = merged_tokens

        return tokens

    def encode(
        self,
        text: str
    ) -> List[int]:

        byte_tokens = (
            self.text_to_bytes(text)
        )

        bpe_tokens = self.apply_bpe(
            byte_tokens
        )

        token_ids = [
            self.special_tokens[
                "<BOS>"
            ]
        ]

        for token in bpe_tokens:

            token_ids.append(
                self.vocab.get(
                    token,
                    self.special_tokens[
                        "<UNK>"
                    ]
                )
            )

        token_ids.append(
            self.special_tokens[
                "<EOS>"
            ]
        )

        return token_ids

    def decode(
        self,
        token_ids: List[int]
    ) -> str:

        byte_stream = b""

        for token_id in token_ids:

            token = (
                self.inverse_vocab.get(
                    token_id
                )
            )

            if token is None:
                continue

            if token in {
                b"<PAD>",
                b"<UNK>",
                b"<BOS>",
                b"<EOS>",
            }:
                continue

            byte_stream += token

        return byte_stream.decode(
            "utf-8",
            errors="replace"
        )

    def save_model(
        self,
        path: str = "tokenizer.model"
    ) -> None:

        model_data = {
            "vocab": self.vocab,
            "inverse_vocab":
                self.inverse_vocab,
            "merges": self.merges,
            "special_tokens":
                self.special_tokens,
            "vocab_size":
                self.vocab_size,
        }

        with open(path, "wb") as f:
            pickle.dump(
                model_data,
                f
            )

        print(
            f"Tokenizer saved to "
            f"{path}"
        )

    def load_model(
        self,
        path: str = "tokenizer.model"
    ) -> None:

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.vocab = model_data["vocab"]
        self.inverse_vocab = (
            model_data["inverse_vocab"]
        )
        self.merges = model_data["merges"]
        self.special_tokens = (
            model_data[
                "special_tokens"
            ]
        )
        self.vocab_size = (
            model_data["vocab_size"]
        )

        print(
            f"Tokenizer loaded from "
            f"{path}"
        )