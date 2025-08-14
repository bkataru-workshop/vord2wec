import re
import random
import math
from collections import defaultdict, Counter

import numpy as np


class MinimalWord2Vec:
    """
    A minimal Word2Vec implementation from scratch using only NumPy.
    This implements the Skip-gram model with negative sampling.
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        window_size: int = 5,
        min_count: int = 5,
        negative_samples: int = 5,
        learning_rate: float = 0.025,
        epochs: int = 10,
    ):
        """
        Initialize the Word2Vec model.

        Args:
            embedding_dim: Dimension of word embeddings (how many numbers represent each word)
            window_size: How many words to consider on each side of the target word
            min_count: Ignore words that appear less than this many times
            negative_samples: Number of negative samples for each positive example
            learning_rate: How fast the model learns (too high = unstable, too low = slow)
            epochs: Number of times to go through the entire dataset
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs

        # These will be initialized during training
        self.word_to_idx = {}  # Maps words to indices
        self.idx_to_word = {}  # Maps indices back to words
        self.word_counts = Counter()  # How often each word appears
        self.vocab_size = 0

        # The actual embedding matrices - these are what we're trying to learn
        self.W_in = None  # Input embeddings (center word representations)
        self.W_out = None  # Output embeddings (context word representations)

        # For negative sampling - words with higher frequency get sampled more often
        self.negative_sampling_table = None

    def preprocess_text(self, text: str) -> list[str]:
        """
        Clean and tokenize the input text.
        This is a simple preprocessing recipe - in practice use a better pipeline pls
        """
