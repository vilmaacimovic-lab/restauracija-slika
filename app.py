import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io
import tempfile
import os

# Podešavanje stranice
st.set_page_config(
    page_title="AI Restauracija i Obnova Starih Slika",
    page_icon="🖼️",
    layout="centered"
)

# Učitavanje Hugging Face tokena u okruženje pre uvoza gradio_client-a
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

from gradio_client import Client, handle_file

# Funkcija za prikaz Adsterra Native Banner reklame
def prikazi_reklamu():
    ad_code = """
    <div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
        <script async fetchpriority="high" data-cfasync="false" src="https://biomanos.org"></script>
        <div id="container-68a59cbef39d33b05863fdf6adf238de"></div>
    </div>
    """
    components.html(ad_code, height=260, scrolling=False)

# Funkcija za uklanjanje žutog tona
def ukloni_zutilo(img_bgr):
    result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# Pred-obrada: uklanja tačkice/mrežicu sa slikanog papira i popravlja kontrast pre AI-ja
def ocisti_mrezicu_i_pripremi(image, skini_mrezicu=True, skini_zuto=True):
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Smanjimo ako je slika ogromna (da AI brže i preciznije obradi)
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 1. Uklanjanje mrežaste teksture (moiré efekta sa papira/ekrana)
    if skini_mrezicu:
        # Median blur skida tačkice, a bilateral filter pegla šum dok čuva ivice
        img_bgr = cv2.medianBlur(img_bgr, 3)
        img_bgr = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=55, sigmaSpace=55)

    # 2. Uklanjanje žutila
    if skini_zuto:
        img_bgr = ukloni_zutilo(img_bgr)

    # 3. Blago podizanje lokalnog kontrasta
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Prava AI restauracija preko CodeFormer modela (sa 4x uvećanjem i jakim AI crtanjem)
def ai_codeformer_restauracija(image, jacina_ai=0.2, uvecanje=4):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        client = Client("sczhou/CodeFormer")
        result = client.predict(
            handle_file(tmp_path),
            True,       # face_align
            True,       # background_enhance
            True,       # face_upsample
            uvecanje,   # upscale (2 ili 4)
            jacina_ai,  # codeformer_fidelity (niže = jače AI crtanje)
            fn_index=0
        )
        while isinstance(result, (list, tuple)):
            result = result
        if isinstance(result, dict) and "path" in result:
            result = result["path"]
        return Image.open(result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# AI Bojenje crno-belih slika preko DDColor modela
def ai_oboji_sliku(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        client = Client("piddnad/DDColor")
        result = client.predict(
            img=handle_file(tmp_path),
            model_size="large",
            input_size=512,
            api_name="/predict"
        )
        while isinstance(result, (list, tuple)):
            result = result
        if isinstance(result, dict) and "path" in result:
            result = result["path"]
        return Image.open(result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- INTERFEJS APLIKACIJE ---
st.title("🖼️ AI Restauracija i Obnova Starih Slika")
st.write("Vratite stari sjaj porodičnim uspomenama – uklonite mrežicu sa papira, izoštrite lica u 4K rezoluciji i obojite crno-bele fotografije.")

prikazi_reklamu()
st.divider()

uploaded_file = st.file_uploader("📂 Izaberite staru fotografiju (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)

    mod = st.radio(
        "Izaberite alat za obnovu:",
        [
            "✨ Dubinska AI Restauracija + Uklanjanje mrežice sa papira (4x HD)",
            "🎨 AI Kolorizacija (Bojenje crno-belih starih slika)",
            "⚡ Brzo čišćenje i uklanjanje žutila (bez AI čekanja)"
        ],
        index=0
    )

    restored_image = None

    if "Dubinska AI" in mod:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            skini_mrezicu = st.checkbox("Ukloni tačkice/mrežicu sa papira (Preporučeno)", value=True)
            skini_zuto = st.checkbox("Neutrališi žuti/izbledeli ton", value=True)
        with col_opt2:
            jacina = st.slider("Jačina AI crtanja (0.1 = najjače, 0.8 = blaže)", 0.1, 0.8, 0.2, step=0.1)
            uvecanje = st.selectbox("Uvećanje rezolucije", [2, 4], index=1)

        if st.button("🚀 Pokreni Pun AI Tretman", type="primary", use_container_width=True):
            with st.spinner("1/2 Čistim mrežicu sa papira... 2/2 AI rekonstruiše lica i detalje (20–40s)..."):
                try:
                    pripremljena = ocisti_mrezicu_i_pripremi(original_image, skini_mrezicu, skini_zuto)
                    ai_rezultat = ai_codeformer_restauracija(pripremljena, jacina_ai=jacina, uvecanje=uvecanje)
                    # Finalno blago izoštravanje
                    ai_rezultat = ImageEnhance.Sharpness(ai_rezultat).enhance(1.25)
                    st.session_state["restored"] = ai_rezultat
                except Exception as e:
                    st.error(f"AI server je trenutno zauzet: {e}")

        if "restored" in st.session_state:
            restored_image = st.session_state["restored"]

    elif "Kolorizacija" in mod:
        st.info("Pretvara stare crno-bele fotografije u prirodne fotografije u boji.")
        if st.button("🎨 Oboji fotografiju pomoću AI-ja", type="primary", use_container_width=True):
            with st.spinner("AI dodaje prirodne boje fotografiji... Sačekajte 15–30 sekundi..."):
                try:
                    pripremljena = ocisti_mrezicu_i_pripremi(original_image, True, False)
                    obojena = ai_oboji_sliku(pripremljena)
                    st.session_state["restored"] = obojena
                except Exception as e:
                    st.error(f"Greška sa serverom za bojenje: {e}")

        if "restored" in st.session_state:
            restored_image = st.session_state["restored"]

    else:
        pripremljena = ocisti_mrezicu_i_pripremi(original_image, True, True)
        restored_image = ImageEnhance.Sharpness(pripremljena).enhance(1.5)

    if restored_image is not None:
        st.success("✅ Fotografija je uspešno obrađena!")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Pre (Original)", use_container_width=True)
        with col2:
            st.image(restored_image, caption="Posle (AI Obnovljena)", use_container_width=True)

        buf = io.BytesIO()
        restored_image.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Preuzmi obnovljenu sliku (HD PNG)",
            data=buf.getvalue(),
            file_name="ai_obnovljena_slika.png",
            mime="image/png",
            use_container_width=True
        )

st.divider()
prikazi_reklamu()
