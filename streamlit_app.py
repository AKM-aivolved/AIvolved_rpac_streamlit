# import streamlit as st

# st.title("🎈 My new old app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )
import subprocess

# Run the bash script
subprocess.run(['bash', 'install_requirements.sh'], check=True)

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from io import BytesIO

def detect_differences(img1, img2, threshold=30):
    # Ensure the images are the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the images
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the differences
    result = img1.copy()
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return result

st.title('Image Difference Detector')

# File uploaders for the two images
uploaded_file1 = st.file_uploader("Choose the first image", type=['png', 'jpg', 'jpeg'])
uploaded_file2 = st.file_uploader("Choose the second image", type=['png', 'jpg', 'jpeg'])

if uploaded_file1 is not None and uploaded_file2 is not None:
    # Create temporary files
    temp_file1 = tempfile.NamedTemporaryFile(delete=False)
    temp_file2 = tempfile.NamedTemporaryFile(delete=False)
    
    # Write the uploaded files to the temporary files
    temp_file1.write(uploaded_file1.read())
    temp_file2.write(uploaded_file2.read())
    
    # Close the temporary files
    temp_file1.close()
    temp_file2.close()
    
    # Read the images using OpenCV
    img1 = cv2.imread(temp_file1.name)
    img2 = cv2.imread(temp_file2.name)
    
    # Delete the temporary files
    os.unlink(temp_file1.name)
    os.unlink(temp_file2.name)
    
    # Convert images from BGR to RGB (OpenCV uses BGR, but Streamlit expects RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Display the original images
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1_rgb, caption='First Image', use_column_width=True)
    with col2:
        st.image(img2_rgb, caption='Second Image', use_column_width=True)
    
    # Add a button to trigger difference detection
    if st.button('Detect Differences'):
        # Perform difference detection
        result = detect_differences(img1, img2)
        
        # Convert result from BGR to RGB
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Display the result
        st.image(result_rgb, caption='Differences Detected', use_column_width=True)

         # Add a download button for the result image
        result_pil = Image.fromarray(result_rgb)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Download Difference Image",
            data=byte_im,
            file_name="difference_image.png",
            mime="image/png"
        )


st.write("Upload two images to detect differences between them.")
