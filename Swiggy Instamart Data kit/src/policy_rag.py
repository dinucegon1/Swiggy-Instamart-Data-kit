"""Policy RAG retrieval and evaluation."""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from mock_llm import chat

class PolicyRAG:
    def __init__(self, policy_dir="data/policies"):
        self.chunks = []
        if os.path.exists(policy_dir):
            for fn in os.listdir(policy_dir):
                if fn.endswith(".txt"):
                    with open(os.path.join(policy_dir, fn), "r", encoding="utf-8") as f:
                        self.chunks.append({"file": fn, "text": f.read().strip()})
        self.vectorizer = TfidfVectorizer()
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
            
    def retrieve(self, query, top_k=1):
        if not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix * q_vec.T).toarray().flatten()
        best_idx = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in best_idx]

def evaluate(rag):
    eval_path = "data/raw/policy_eval.csv"
    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)
    else:
        df = pd.DataFrame(columns=["question", "expected_doc", "expected_fact"])
    results = []
    for _, row in df.iterrows():
        res = rag.retrieve(row["question"])
        doc = res[0]["file"] if res else "NONE"
        results.append({"question": row["question"], "retrieved_doc": doc, "match": doc == row["expected_doc"]})
    return pd.DataFrame(results)
