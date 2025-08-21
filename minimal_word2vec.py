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

    def preprocess_text(self, text: str) -> [str]:
        """
        Clean and tokenize the input text.
        This is a simple preprocessing recipe - in practice use a better pipeline pls
        """
        # Convert to lowercase and remove non-alphabetic characters
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())

        # Split into words
        words = text.split()
        return words

    def build_vocabulary(self, sentences: [[str]]):
        """
        Build vocabulary from the training sentences.
        Only keeps words that appear at least min_count times.
        """
        # Count how often each word appears
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1

        # Filter out rare words
        filtered_words = [
            word for word, count in self.word_counts.items() if count >= self.min_count
        ]

        # Create mappings between words and indices
        self.word_to_idx = {word: idx for idx, word in enumerate(filtered_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(filtered_words)

        print(f"Vocabulary size: {self.vocab_size} words")
        print(f"Most common words: {self.word_counts.most_common(10)}")

    def initialize_embeddings(self):
        """
        Initialize the embedding matrices with random small values.
        We use two matrices: one for input (center words) and one for output (context words).
        """
        # Initialize with small random values - this is crucial for training
        self.W_in = (
            np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))
            / self.embedding_dim
        )
        self.W_out = (
            np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim))
            / self.embedding_dim
        )

    def create_negative_sampling_table(self, table_size: int = 100000):
        """
        Create a table for efficient negative sampling.
        Words with higher frequency appear more often in the table.
        Uses the word2vec paper's formula: count^(3/4) for subsampling frequent words.
        """
        # Calculate sampling probabilities
        total_count = sum(self.word_counts.values())
        word_probs = []

        for word in self.idx_to_word.values():
            # Use fractional power to reduce probability of very common words
            prob = (self.word_counts[word] / total_count) ** 0.75
            word_probs.append(prob)

        # Normalize probabilities
        word_probs = np.array(word_probs)
        word_probs = word_probs / word_probs.sum()

        # Create the sampling table
        self.negative_sampling_table = []
        for i, prob in enumerate(word_probs):
            count = int(prob * table_size)
            self.negative_sampling_table.extend([i] * count)

        # Handle rounding errors
        while len(self.negative_sampling_table) < table_size:
            self.negative_sampling_table.append(random.choice(range(self.vocab_size)))

    def get_negative_samples(self, positive_idx: int) -> [int]:
        """
        Sample negative examples (words that don't actually appear in context).
        Make sure we don't sample the positive example itself.
        """
        negative_samples = []
        while len(negative_samples) < self.negative_samples:
            sample = random.choice(self.negative_sampling_table)
            if sample != positive_idx and sample not in negative_samples:
                negative_samples.append(sample)
        return negative_samples

    def sigmoid(self, x):
        """
        Sigmoid function with numerical stability.
        """
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def train_pair(self, center_idx: int, context_idx: int):
        """
        Train on a single (center_word, context_word) pair using negative sampling.
        This is where the actual learning happens!
        """
        # Get embeddings for center and context words
        center_embedding = self.W_in[center_idx]  # Shape: (embedding_dim,)
        context_embedding = self.W_out[context_idx]  # Shape: (embedding_dim,)

        # Positive example: center and context actually appear together
        # We want this to have high probability (close to 1)
        score = np.dot(center_embedding, context_embedding)
        pred = self.sigmoid(score)

        # Calculate gradient for positive example
        # If prediction is wrong, gradient will be larger
        positive_grad = (1 - pred) * self.learning_rate

        # Update embeddings for positive example
        self.W_out[context_idx] += positive_grad * center_embedding
        center_grad = positive_grad * context_embedding

        # Negative examples: these pairs don't actually appear together
        # We want these to have low probability (close to 0)
        negative_samples = self.get_negative_samples(context_idx)

        for neg_idx in negative_samples:
            neg_embedding = self.W_out[neg_idx]
            neg_score = np.dot(center_embedding, neg_embedding)
            neg_pred = self.sigmoid(neg_score)

            # For negative examples, we want prediction to be 0
            negative_grad = -neg_pred * self.learning_rate

            # Update embeddings for negative example
            self.W_out[neg_idx] += negative_grad * center_embedding
            center_grad += negative_grad * neg_embedding

        # Update center word embedding
        self.W_in[center_idx] += center_grad

    def generate_training_pairs(self, sentences: [[str]]):
        """
        Generate (center_word, context_word) training pairs from sentences.
        For each word, we look at surrounding words within the window.
        """
        pairs = []
        for sentence in sentences:
            # Convert words to indices, skip unknown words
            word_indices = []
            for word in sentence:
                if word in self.word_to_idx:
                    word_indices.append(self.word_to_idx[word])

            # Generate pairs for each word in the sentence
            for i, center_idx in enumerate(word_indices):
                # Look at surrounding words within the window
                for j in range(
                    max(0, i - self.window_size),
                    min(len(word_indices), i + self.window_size + 1),
                ):
                    if i != j:  # Don't pair a word with itself
                        context_idx = word_indices[j]
                        pairs.append((center_idx, context_idx))

        return pairs

    def train(self, text: str):
        """
        Main training function. This orchestrates the entire training process.
        """
        print("Preprocessing text...")
        words = self.preprocess_text(text)

        # Split into sentences (simple approach, in practice you might want better sentence splitting)
        sentences = []
        current_sentence = []
        for word in words:
            current_sentence.append(word)
            # Simple sentence boundary detection
            if len(current_sentence) > 20:  # Arbitrary sentence length
                sentences.append(current_sentence)
                current_sentence = []
        if current_sentence:  # Don't forget the last sentence
            sentences.append(current_sentence)

        print("Building vocabulary...")
        self.build_vocabulary(sentences)

        if self.vocab_size == 0:
            raise ValueError(
                "No words in vocabulary! Check your text and min_count parameter."
            )

        print("Initializing embeddings...")
        self.initialize_embeddings()
        self.create_negative_sampling_table()

        print("Generating training pairs...")
        training_pairs = self.generate_training_pairs(sentences)
        print(f"Generated {len(training_pairs)} training pairs")

        print(f"Training for {self.epochs} epochs...")

        # Training loop
        for epoch in range(self.epochs):
            # Shuffle training pairs for better learning
            random.shuffle(training_pairs)

            # Train on each pair
            for i, (center_idx, context_idx) in enumerate(training_pairs):
                self.train_pair(center_idx, context_idx)

                # Progress update
                if i % 10000 == 0 and i > 0:
                    print(
                        f"Epoch {epoch + 1}/{self.epochs}, Pair {i}/{len(training_pairs)}"
                    )

            # Decay learning rate over time
            self.learning_rate *= 0.95
            print(f"Completed epoch {epoch + 1}/{self.epochs}")

        print("Training completed!")

    def get_word_vector(self, word: str) -> np.ndarray:
        """Get the embedding vector for a word."""
        if word not in self.word_to_idx:
            raise KeyError(f"Word '{word}' not in vocabulary")
        idx = self.word_to_idx[word]
        return self.W_in[idx]  # Use input embeddings as final word vectors

    def most_similar(self, word: str, top_k: int = 10):
        """
        Find the most similar words to the given word using cosine similarity.
        This is where you can see if the model learned meaningful relationships!
        """
        if word not in self.word_to_idx:
            print(f"Word '{word}' not in vocabulary")
            return []

        word_vector = self.get_word_vector(word)

        # Calculate cosine similarities with all other words
        similarities = []
        for other_word, idx in self.word_to_idx.items():
            if other_word == word:
                continue

            other_vector = self.W_in[idx]

            # Cosine similarity = dot product / (magnitude1 * magnitude2)
            similarity = np.dot(word_vector, other_vector) / (
                np.linalg.norm(word_vector) * np.linalg.norm(other_vector)
            )
            similarities.append((other_word, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Example usage and demonstration
if __name__ == "__main__":
    # Sample text for demonstration
    sample_text = (
        """
    The king sat on his throne in the royal palace. The queen walked through the castle gardens.
    A man walked his dog in the park. The woman fed her cat some fish.
    The prince rode his horse to the village. The princess danced at the royal ball.
    Dogs and cats are popular pets. Many people love their pets very much.
    The royal family lives in a magnificent castle. The king and queen rule the kingdom wisely.
    Fish swim in the ocean. Birds fly in the sky. Animals live in the forest.
    The man and woman went to the market. They bought food for their family.
    Children play in the playground. Adults work in the office.
    The sun shines brightly in the sky. The moon glows softly at night.
    Water flows in the river. Trees grow in the forest.
    """
        * 50
    )  # Repeat to have more training data

    # Create and train the model
    model = MinimalWord2Vec(
        embedding_dim=50,  # Smaller for demo
        window_size=3,  # Look at 3 words on each side
        min_count=3,  # Need at least 3 occurances
        negative_samples=5,
        learning_rate=0.025,
        epochs=5,
    )

    # Train the model
    model.train(sample_text)

    # Test the model
    print("\n" + "=" * 50)
    print("TESTING THE MODEL")
    print("=" * 50)

    test_words = ["king", "dog", "main", "castle"]
    for word in test_words:
        try:
            print(f"\nMost similar words to '{word}':")
            similar_words = model.most_similar(word, top_k=5)
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
        except KeyError as e:
            print(f"  {e}")
