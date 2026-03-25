from sentence_transformers import SentenceTransformer

_MODEL = None


def get_sentence_embedding(sentence):
    # Force CPU to avoid CUDA conflicts with the main LLM on the accelerator.
    # Model is loaded once and reused for all subsequent calls.
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    return _MODEL.encode(sentence)
