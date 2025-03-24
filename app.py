import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import os

# Load Stable Diffusion model
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# Streamlit UI
st.title("👗 AI Fashion Model Visualization Tool")
st.write("Upload your photo and generate AI-powered fashion outfits.")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # User input for fashion style
    prompt = st.text_input("Describe your fashion style (e.g., 'red dress with floral patterns')", "A trendy black leather jacket with gold zippers")

    if st.button("Generate AI Outfit"):
        with st.spinner("Generating outfit..."):
            # Run Stable Diffusion
            image = pipe(prompt).images[0]
            image_path = "generated_outfit.png"
            image.save(image_path)

            # Display result
            st.image(image, caption="Generated AI Outfit", use_column_width=True)
            st.success("Fashion Model Generated!")

            # Download option
            with open(image_path, "rb") as file:
                btn = st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="AI_Fashion_Model.png",
                    mime="image/png"
                )
