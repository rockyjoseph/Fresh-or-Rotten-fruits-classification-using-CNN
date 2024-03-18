import numpy as np

import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps

from src.pipeline.predict_pipeline import PredictPipeline

LABELS = ['Fresh Peaches','Fresh Pomegranates','Fresh Strawberries',
            'Rotten Peaches','Rotten Pomegranates','Rotten Strawberries']

model = PredictPipeline()

st.write("# Fruits Classification")

file = st.file_uploader("Choose File", type=['jpg','png','jpeg'])


if file is None:
    st.text("Please upload an Image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])

    predicted_label = LABELS[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.header("This image most likely belongs to {} with a {:.2f} percent confidence.".format(predicted_label, confidence))