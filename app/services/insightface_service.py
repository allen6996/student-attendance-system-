import insightface

class InsightFaceService:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_and_embed(self, frame):
        faces = self.app.get(frame)
        if not faces:
            return None, None

        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        best_face = faces[0]
        embedding = best_face.normed_embedding
        return best_face, embedding.tolist()
        # 🔥 NEW FUNCTION FOR MULTI-FACE DETECTION
    def detect_all_faces(self, frame):
        faces = self.app.get(frame)

        results = []
        for face in faces:
            embedding = face.embedding
            results.append((face, embedding))

        return results

