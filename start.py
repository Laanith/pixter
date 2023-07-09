import streamlit as st
from transformers import VisionEncoderDecoderModel
from transformers import ViTFeatureExtractor
from transformers import AutoTokenizer
import torch
from PIL import Image
import warnings


warnings.filterwarnings('ignore')


model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(
        images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


# st.sidebar.header('Navigation')
st.sidebar.subheader('Navigation')
nav = st.sidebar.radio(' ', ['Home', 'About'])
if nav == 'Home':
    st.title('Welcome to Pixter,')
    st.subheader('a one stop destination for image manipulation tools')
    st.image('./Artificial.jpg', width=800)
    st.header('Try out our AI image caption generator')
    img = st.file_uploader('Upload your image here')
    if img:
        st.text('Here is the image you uploaded')
        lis = predict_step([img])[0]
        st.image(img, width=200)
        st.text(lis)


else:
    st.title('About Us')
    st.text('This project is done by a team of two')
    st.text(' students,')
    st.text('Laanith chouhan and Thanish.')
