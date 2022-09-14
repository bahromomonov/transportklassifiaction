import posixpath
import streamlit as st
import pathlib
temp=pathlib.posixpath
pathlib.PosixPath=pathlib.WindowsPath
from fastai.vision.all import *

st.title('Transportlarni klassifikatsiyalovchi model')

file=st.file_uploader('Rasm yuklash',type=['jpeg','jpg','svg'])
if file:
    st.image(file)
    img=PILImage.create(file)
    model=load_learner('transport_model.pkl')
    pred,pred_id,probs=model.predict(img)
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik:{probs[pred_id]*100:1f}%")
   
