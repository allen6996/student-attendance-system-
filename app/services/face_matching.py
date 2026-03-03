import numpy as np

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_best_match(input_emb, stored_embeddings, threshold=0.40):
    best_score = -1
    best_student_id = None

    for item in stored_embeddings:
        score = cosine_similarity(input_emb, item["embedding"])
        if score > best_score:
            best_score = score
            best_student_id = item["student_id"]

    if best_score >= threshold:
        return best_student_id, best_score

    return None, best_score
