import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Seite konfigurieren (Titel im Browser-Tab)
st.set_page_config(page_title="Strator Vision V6", layout="wide")

st.title("Strator Vision AI – V6 Prototype")
st.markdown("---")

# Seitenleiste für die Steuerung
st.sidebar.header("Veredelungs-Einstellungen")
strength = st.sidebar.slider("Schärfegrad (Textur)", 1.0, 2.5, 1.3, 0.1)
denoise_val = st.sidebar.slider("Entrauschung (Luminanz)", 5, 25, 12)

# Dateiupload
uploaded_file = st.file_uploader("Lade ein Mode-Foto hoch (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Bild verarbeiten
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Layout: Zwei Spalten für den Vergleich
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Asset")
        st.image(img_rgb, use_container_width=True)
    
    # --- STRATOR V6 ENGINE ---
    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        results = segmenter.process(img_rgb)
        mask = results.segmentation_mask > 0.5
        
        # LAB-Farbraum für Luminanz-Schutz
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Entrauschen
        l_denoised = cv2.fastNlMeansDenoising(l, None, denoise_val, 7, 21)
        
        # Schärfen
        gaussian = cv2.GaussianBlur(l_denoised, (0, 0), 2)
        l_sharp = cv2.addWeighted(l_denoised, strength, gaussian, -(strength-1), 0)
        
        # Rekonstruktion
        enhanced_img = cv2.cvtColor(cv2.merge((l_sharp, a, b)), cv2.COLOR_LAB2RGB)
        
        # Semantic Body Guard (Hautschutz)
        final_img = np.copy(enhanced_img)
        final_img[mask] = img_rgb[mask]
        
    with col2:
        st.subheader("V6 Veredelt")
        st.image(final_img, use_container_width=True)
        st.success("KI-Schutz aktiv: Biometrische Daten (Haut) im Original erhalten.")

else:
    st.info("Bitte lade ein Bild hoch, um die V6-Engine zu starten.")
