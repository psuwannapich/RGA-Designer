from sentence_transformers import SentenceTransformer


def get_sentence_embedding(sentence):
    # Force CPU to avoid CUDA conflicts with the main LLM on the accelerator.
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(sentence)
    return embeddings
