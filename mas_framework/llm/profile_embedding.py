from sentence_transformers import SentenceTransformer

_MODEL = None

_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def get_sentence_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it once on first call."""
    global _MODEL
    if _MODEL is None:
        # Force CPU: tiny model (~90 MB), avoids CUDA conflicts with the main LLM.
        _MODEL = SentenceTransformer(_MODEL_NAME, device='cpu')
    return _MODEL


def get_sentence_embedding(sentence):
    return get_sentence_model().encode(sentence)
