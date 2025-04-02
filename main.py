import streamlit as st
from PIL import Image
from alpha_prototype import AlphaPrototype

def main():
    st.title("Face Recognition App with DeepFace")

    # Create system instance
    system = AlphaPrototype(model_name="VGG-Face")
    msg = system.initialize(force_reinitialize=True)
    st.success(msg)

    st.header("1. Identify a Person from Uploaded Image")
    uploaded_image = st.file_uploader("Upload an image for recognition", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)
        temp_path = "temp_query.jpg"
        img.save(temp_path)

        name, dist = system.find_match(temp_path)
        st.write("Predicted Identity:", " ".join(name.split("_")))
        if dist is not None:
            st.write("Cosine Similarity:", round(dist, 3))

    st.header("2. Add a New Face to the Database")
    new_name = st.text_input("Enter the name of the person")
    new_name = "_".join(new_name.split())
    new_image = st.file_uploader("Upload an image to add to the database", type=["jpg", "jpeg", "png"], key="add_face")
    if new_image and new_name:
        img = Image.open(new_image).convert("RGB")
        st.image(img, caption="Image to Add", width=300)
        if st.button("Add to Database"):
            msg = system.add_to_db(img, new_name)
            if msg.startswith("Error"):
                st.error(msg)
            else:
                st.success(msg)

if __name__ == "__main__":
    main()