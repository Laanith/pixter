import streamlit as st
import script as scr

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
        lis = scr.predict_step([img])[0]
        st.image(img, width=200)
        st.text(lis)


else:
    st.title('About Us')
    st.text('This project is done by a team of two')
    st.text(' students,')
    st.text('Laanith chouhan and Thanish.')
