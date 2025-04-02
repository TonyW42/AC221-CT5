import os
import numpy as np
import pickle
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

class AlphaPrototype:
    def __init__(self, model_name="VGG-Face"):
        self.model_name = model_name
        self.template_db = "template_db"
        self.embed_file = os.path.join(self.template_db, "embeddings.npy")
        self.meta_file = os.path.join(self.template_db, "metadata.pkl")

    def get_embedding_size(self):
        sizes = {
            "VGG-Face": 2622,  # Updated to match your DeepFace version
            "Facenet": 128,
            "Facenet512": 512,
            "OpenFace": 128,
            "DeepFace": 4096,
            "ArcFace": 512,
            "Dlib": 128,
            "SFace": 128
        }
        return sizes[self.model_name]

    def initialize(self, force_reinitialize=False):
        os.makedirs(self.template_db, exist_ok=True)
        embedding_size = self.get_embedding_size()

        if not force_reinitialize and os.path.exists(self.embed_file) and os.path.exists(self.meta_file):
            return f"Database already initialized (model: {self.model_name})"

        all_embeddings = []
        all_metadata = []

        # Debug: Try to get one embedding to check size
        test_files = []
        for person_name in os.listdir(self.template_db):
            person_dir = os.path.join(self.template_db, person_name)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        test_files.append(os.path.join(person_dir, img_name))
                        break
                if test_files:
                    break

        if test_files:
            try:
                test_embedding = DeepFace.represent(
                    img_path=test_files[0],
                    model_name=self.model_name,
                    enforce_detection=False
                )[0]["embedding"]
                print(f"DEBUG: Actual embedding size from DeepFace: {len(test_embedding)}")
                embedding_size = len(test_embedding)  # Use actual size instead of predefined
            except Exception as e:
                print(f"DEBUG: Failed to get test embedding: {e}")

        for person_name in os.listdir(self.template_db):
            person_dir = os.path.join(self.template_db, person_name)
            if not os.path.isdir(person_dir):
                continue

            for img_name in os.listdir(person_dir):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(person_dir, img_name)
                try:
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name=self.model_name,
                        enforce_detection=False
                    )[0]["embedding"]
                    all_embeddings.append(embedding)
                    all_metadata.append({"name": person_name, "path": img_path})
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")

        if not all_embeddings:
            return "Initialization failed: No valid embeddings found."

        all_embeddings = np.array(all_embeddings)
        np.save(self.embed_file, all_embeddings)
        with open(self.meta_file, "wb") as f:
            pickle.dump(all_metadata, f)

        return f"Initialized face database with {len(all_metadata)} images (model: {self.model_name})"

    def find_match(self, uploaded_image_path, threshold=0.7):
        if not os.path.exists(self.embed_file) or not os.path.exists(self.meta_file):
            return "Database not initialized", None

        try:
            query_embedding = DeepFace.represent(
                img_path=uploaded_image_path,
                model_name=self.model_name,
                enforce_detection=False
            )[0]["embedding"]
            query_embedding = np.array(query_embedding).reshape(1, -1)

            db_embeddings = np.load(self.embed_file)
            with open(self.meta_file, "rb") as f:
                metadata = pickle.load(f)

            if db_embeddings.shape[0] == 0:
                return "Unrecognized", None

            if db_embeddings.shape[1] != query_embedding.shape[1]:
                return f"Model mismatch: DB has shape {db_embeddings.shape[1]}, query has {query_embedding.shape[1]}", None

            sims = cosine_similarity(query_embedding, db_embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            if best_score >= (1 - threshold):
                return metadata[best_idx]["name"], best_score
            else:
                return "Unrecognized", best_score

        except Exception as e:
            print(f"Error processing query image: {e}")
            return f"Error: {str(e)}", None

    def add_to_db(self, image, name):
        person_dir = os.path.join(self.template_db, name)
        os.makedirs(person_dir, exist_ok=True)
        img_count = len(os.listdir(person_dir))
        img_path = os.path.join(person_dir, f"{name}_{img_count:04d}.jpg")
        image.save(img_path)

        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                enforce_detection=False
            )[0]["embedding"]
            embedding = np.array(embedding).reshape(1, -1)

            if os.path.exists(self.embed_file) and os.path.exists(self.meta_file):
                db_embeddings = np.load(self.embed_file)
                with open(self.meta_file, "rb") as f:
                    metadata = pickle.load(f)

                if db_embeddings.shape[1] != embedding.shape[1]:
                    return f"Error: Model mismatch. Database uses embeddings of shape {db_embeddings.shape[1]}, but new embedding has shape {embedding.shape[1]}"

                db_embeddings = np.vstack([db_embeddings, embedding])
                metadata.append({"name": name, "path": img_path})
            else:
                db_embeddings = embedding
                metadata = [{"name": name, "path": img_path}]

            np.save(self.embed_file, db_embeddings)
            with open(self.meta_file, "wb") as f:
                pickle.dump(metadata, f)

            return f"Image and embedding added for {name}."
        except Exception as e:
            return f"Error: {str(e)}"
