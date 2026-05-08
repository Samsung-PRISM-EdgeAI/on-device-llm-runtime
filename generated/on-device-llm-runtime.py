import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import math

# --- Dummy Model and Tokenizer for demonstration ---
class DummyTokenizer:
    """A dummy tokenizer for demonstration purposes."""
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        # Simple hash-based encoding for demonstration
        # Map chars to vocab, reserving 0,1,2 for special tokens
        tokens = [ord(c) % (self.vocab_size - 3) + 3 for c in text]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        # Simple reverse mapping
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.eos_token_id, self.bos_token_id]]
        # Map