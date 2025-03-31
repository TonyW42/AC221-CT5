import os
import numpy as np
import pickle
from deepface import DeepFace
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Default model
MODEL_NAME = "VGG-Face"

DB_DIR = "face_db"
EMBED_FILE = os.path.join(DB_DIR, "embeddings.npy")
META_FILE = os.path.join(DB_DIR, "metadata.pkl")

# Get embedding size from model
def get_embedding_size(model_name):
    # Based on DeepFace's model specs
    return {
        "VGG-Face": 4096,
        "Facenet": 128,
        "Facenet512": 512,
        "OpenFace": 128,
        "DeepFace": 4096,
        "ArcFace": 512,
        "Dlib": 128,
        "SFace": 128
    }[model_name]

# Initialize an empty database
def initialize_db(model_name=MODEL_NAME):
    os.makedirs(DB_DIR, exist_ok=True)

    embedding_size = get_embedding_size(model_name)
    if not os.path.exists(EMBED_FILE):
        np.save(EMBED_FILE, np.empty((0, embedding_size)))
    if not os.path.exists(META_FILE):
        with open(META_FILE, "wb") as f:
            pickle.dump([], f)
    return f"Initialized empty face database (model: {model_name})"

# Add image and its embedding to DB
def add_to_db(image, name, model_name=MODEL_NAME):
    person_dir = os.path.join(DB_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    img_count = len(os.listdir(person_dir))
    img_path = os.path.join(person_dir, f"{name}_{img_count}.jpg")
    image.save(img_path)

    # Get embedding
    embedding = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=False)[0]["embedding"]
    embedding = np.array(embedding).reshape(1, -1)

    # Load DB
    db_embeddings = np.load(EMBED_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)

    # Check if dimensions match
    if db_embeddings.shape[1] != embedding.shape[1]:
        return f"Error: Model mismatch. Database uses embeddings of shape {db_embeddings.shape[1]}, but new embedding has shape {embedding.shape[1]}"

    # Append and save
    db_embeddings = np.vstack([db_embeddings, embedding])
    metadata.append({"name": name, "path": img_path})

    np.save(EMBED_FILE, db_embeddings)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    return f"Image and embedding added for {name}."

# Find best match from precomputed embeddings
def find_match(uploaded_image_path, model_name=MODEL_NAME, threshold=0.75):
    if not os.path.exists(EMBED_FILE) or not os.path.exists(META_FILE):
        return "Database not initialized", None

    query_embedding = DeepFace.represent(
        img_path=uploaded_image_path,
        model_name=model_name,
        enforce_detection=False
    )[0]["embedding"]
    query_embedding = np.array(query_embedding).reshape(1, -1)

    db_embeddings = np.load(EMBED_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)

    if db_embeddings.shape[0] == 0:
        return "Unrecognized", None

    if db_embeddings.shape[1] != query_embedding.shape[1]:
        return f"Model mismatch: DB has shape {db_embeddings.shape[1]}, query has {query_embedding.shape[1]}", None

    sims = cosine_similarity(query_embedding, db_embeddings)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score >= (1 - threshold):
        return metadata[best_idx]["name"], 1 - best_score
    else:
        return "Unrecognized", 1 - best_score

# Streamlit App
def main():
    st.title("Face Recognition App with DeepFace")

    msg = initialize_db()
    st.success(msg)

    st.header("1. Identify a Person from Uploaded Image")
    uploaded_image = st.file_uploader("Upload an image for recognition", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)
        temp_path = "temp_query.jpg"
        img.save(temp_path)

        name, dist = find_match(temp_path)
        st.write("Predicted Identity:", name)
        if dist is not None:
            st.write("Cosine Distance:", round(dist, 3))

    st.header("2. Add a New Face to the Database")
    new_name = st.text_input("Enter the name of the person")
    new_image = st.file_uploader("Upload an image to add to the database", type=["jpg", "jpeg", "png"], key="add_face")
    if new_image and new_name:
        img = Image.open(new_image).convert("RGB")
        st.image(img, caption="Image to Add", width=300)
        if st.button("Add to Database"):
            msg = add_to_db(img, new_name)
            if msg.startswith("Error"):
                st.error(msg)
            else:
                st.success(msg)

if __name__ == "__main__":
    main()
