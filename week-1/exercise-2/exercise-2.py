import streamlit as st
import numpy as np
from PIL import Image
import gdown

IMG_NAME = "default_1.png"

with st.sidebar:
    st.write("Upload an image")
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.write("Or select a default image")
    btn_default = st.button("Default Image")

    st.write("Process methods")
    process_method = st.selectbox(
        "Select a method",
        ("Lightness", "Average", "Luminosity"))
    btn_process = st.button("Process image")

img = None
if btn_default:
    img = np.array(Image.open(IMG_NAME))
    st.image(img, caption="Default image", use_column_width=True)
    img = img.astype(np.float16)


if btn_process:
    if upload is not None:
        img = np.array(Image.open(upload))
        st.image(img, caption="Uploaded image", use_column_width=True)
        img = img.astype(np.float16)
    else:
        img = np.array(Image.open(IMG_NAME))
        st.image(img, caption="Default image", use_column_width=True)
        img = img.astype(np.float16)

    if process_method == "Lightness":
        process_img = np.apply_along_axis(
            lambda x: (max(x) + min(x)) / 2,
            axis=2,
            arr=img
        )
    elif process_method == "Average":
        process_img = np.apply_along_axis(
            lambda x: (x[0] + x[1] + x[2]) / 3,
            axis=2,
            arr=img
        )
    else:
        process_img = np.apply_along_axis(
            lambda x: (x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722),
            axis=2,
            arr=img
        )

    process_img = process_img.astype(np.uint8)
    st.image(process_img, caption="Processed image", use_column_width=True)
    st.write(f"process_img[0,0] = {process_img[0, 0]}")
