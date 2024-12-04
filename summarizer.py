import re
import ssl
import numpy as np
import nltk
from typing import List, Optional

try:
    import spacy
except ImportError:
    spacy = None

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize
except ImportError:
    nltk = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextSummarizer:
    def __init__(self, use_spacy: bool = False):
        """
        Initialize the text summarizer with optional advanced tokenization

        :param use_spacy: Whether to use spaCy for advanced tokenization
        """
        # Handle NLTK data path and SSL certificate
        self._setup_nltk()

        # Choose tokenization method
        self.nlp = None
        self.use_spacy = use_spacy and spacy is not None
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"SpaCy model load failed: {e}")
                self.use_spacy = False

        # Set up stopwords
        self.stop_words = self._get_stopwords()

    def _setup_nltk(self):
        """
        Set up NLTK with SSL certificate workaround
        """
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Specific to your NLTK data path
        nltk.data.path.append('/Users/jesseelorddushime/nltk_data')

    def _get_stopwords(self) -> set:
        """
        Retrieve stopwords with fallback mechanism

        :return: Set of stopwords
        """
        fallback_stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with'
        }

        try:
            return set(stopwords.words('english'))
        except Exception as e:
            print(f"NLTK stopwords load failed: {e}")
            return fallback_stop_words

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing

        :param text: Input text
        :return: Cleaned and normalized text
        """
        # Remove unnecessary whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Correct common punctuation issues
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)

        # Remove multiple spaces after punctuation
        text = re.sub(r'([.!?])\s+', r'\1 ', text)

        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences with advanced handling

        :param text: Input text
        :return: List of sentences
        """
        # Preprocess text first
        text = self.preprocess_text(text)

        # Use SpaCy if available
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Fallback to NLTK
        try:
            return sent_tokenize(text)
        except Exception:
            # Simple fallback tokenization
            return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

    def _compute_sentence_length_weight(self, sentences: List[str]) -> np.ndarray:
        """
        Compute sentence length weights

        :param sentences: List of sentences
        :return: Array of length weights
        """
        # Compute sentence lengths
        lengths = np.array([len(sentence.split()) for sentence in sentences])

        # Normalize lengths
        max_length = np.max(lengths)
        normalized_lengths = lengths / max_length if max_length > 0 else lengths

        return normalized_lengths

    def summarize(self, text: str, num_sentences: int = 3,
                  use_length_weight: bool = True) -> str:
        """
        Generate a summary of the text

        :param text: Input text to summarize
        :param num_sentences: Number of sentences in summary
        :param use_length_weight: Whether to use sentence length weighting
        :return: Summary text
        """
        # Handle empty or very short text
        if not text or len(text.split()) < 10:
            return text

        # Tokenize sentences
        sentences = self.tokenize_sentences(text)

        # If text is too short, return original text
        if len(sentences) <= num_sentences:
            return text

        try:
            # Create TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')

            # Convert sentences to TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Compute sentence scores
            sentence_scores = np.zeros(len(sentences))
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        sentence_scores[i] += similarity_matrix[i][j]

            # Apply length weighting if enabled
            if use_length_weight:
                length_weights = self._compute_sentence_length_weight(sentences)
                sentence_scores *= length_weights

            # Get top sentences
            top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_sentence_indices.sort()

            # Create summary
            summary = ' '.join([sentences[i] for i in top_sentence_indices])

            return summary

        except Exception as e:
            print(f"Summary generation error: {e}")
            # Fallback: return first few sentences
            return ' '.join(sentences[:num_sentences])