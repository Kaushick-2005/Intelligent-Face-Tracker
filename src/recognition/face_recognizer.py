import os

# ------------------------------------------------------------
# Force InsightFace cache/models directory to D-Drive
# ------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["INSIGHTFACE_HOME"] = os.path.join(ROOT_DIR, "models", "insightface")

import cv2
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self, ctx_id=0 if torch.cuda.is_available() else -1):
        """
        Initializes InsightFace FaceAnalysis module.
        ctx_id: 0 for GPU, -1 for CPU
        """
        print("[INFO] Initializing InsightFace Recognition Engine...")
        print("[INFO] Model storage directory:", os.environ["INSIGHTFACE_HOME"])

        self.app = FaceAnalysis(
            name='buffalo_l',
            root=os.environ["INSIGHTFACE_HOME"],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id == 0 else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("[INFO] InsightFace initialized successfully.")

    def extract_embedding(self, face_crop):
        """
        Extracts 512-D embedding from a cropped face image.
        Returns numpy array embedding or None if extraction fails.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        faces = self.app.get(face_crop)
        if not faces:
            return None

        return faces[0].embedding

    @staticmethod
    def compute_similarity(emb1, emb2):
        """
        Computes Cosine Similarity between two face embeddings.
        Result ranges from -1.0 to 1.0 (Higher = More Similar).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        first = np.asarray(emb1, dtype=np.float32).reshape(-1)
        second = np.asarray(emb2, dtype=np.float32).reshape(-1)
        if first.shape != second.shape or not (
            np.all(np.isfinite(first)) and np.all(np.isfinite(second))
        ):
            return 0.0
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator <= np.finfo(np.float32).eps:
            return 0.0
        similarity = float(np.dot(first, second) / denominator)
        return similarity if np.isfinite(similarity) else 0.0


if __name__ == "__main__":
    recognizer = FaceRecognizer()
    print("Recognizer module ready on D-Drive.")