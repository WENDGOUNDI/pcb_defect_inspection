
import streamlit as st
import PIL
from super_gradients.training import models
import pandas as pd
import numpy as np
from collections import Counter

# Page Layout

st.set_page_config(
    page_title="Defect Inspection",
    #page_icon = "",
    layout="wide",
    initial_sidebar_state = "expanded"
)


# Creating sidebar
with st.sidebar:
    st.header("Upload Image")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 50)) / 100 # The default confidence score is set to 50

# Creating main page heading
st.title("PCB DEFECT INSPECTION WEBAPP")

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image,
                 caption="Uploaded Image",
                 use_column_width=True

                 )

try:
    custom_model_path = "checkpoints/Train_2/ckpt_best.pth"
    best_model = models.get('yolo_nas_l',
                            num_classes=2,
                            checkpoint_path= custom_model_path,
                            )

except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {custom_model_path}")
    st.error(ex)

if st.sidebar.button('INSPECT'):
    best_model.predict(uploaded_image, conf=0.5, ).save("output_folder")
    predictions = best_model.predict(uploaded_image, conf=0.5)
    prediction_objects = list(predictions._images_prediction_lst)[0]
    bboxes = prediction_objects.prediction.bboxes_xyxy

    int_labels = prediction_objects.prediction.labels.astype(int)
    class_names = prediction_objects.class_names
    pred_classes = [class_names[i] for i in int_labels]
    count_res = dict(Counter(pred_classes))
    chart_data = pd.DataFrame([count_res])

    with col2:
        predicted_image = PIL.Image.open("output_folder/pred_0.jpg")
        st.image(predicted_image,
                 caption='Predicted Image',
                 use_column_width=True
                 )

    st.header("INSPECTION RATIO")
    st.bar_chart(chart_data)


