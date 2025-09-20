class VectorStoreWrapper:
    def __init__(self, kind="chroma"):
        self.kind = kind
    def add(self, embeddings, metadata=None):
        return True
    def search(self, query_vec, k=5):
        return []