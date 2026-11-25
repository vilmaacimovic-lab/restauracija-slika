import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io
import tempfile
import os
from gradio_client import Client, handle_file

# Podešavanje stranice
st.set_page_config(
    page_title="AI Restauracija i Obnova Starih Slika",
    page_icon="🖼️",
    layout="centered"
)

# Funkcija za prikaz Adsterra Native Banner reklame
def prikazi_reklamu():
    ad_code = """
    <div style="display: flex; justify-content: center; align-items: center; margin: 10px 0;">
        <script async fetchpriority="high" data-cfasync="false" src="https://biomanos.org"></script>
        <div id="container-68a59cbef39d33b05863fdf6adf238de"></div>
    </div>
    """
    components.html(ad_code, height=260, scrolling=False)

# Funkcija za uklanjanje žutog tona (Auto White Balance)
def ukloni_zutilo(img_bgr):
    result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

# Brza lokalna obrada
def brza_restauracija(image, ukloni_sum, neutralisi_zuto, popravi_kontrast, ostrina, crno_belo):
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. Neutralisanje žutog tona
    if neutralisi_zuto:
        img_bgr = ukloni_zutilo(img_bgr)

    # 2. Uklanjanje šuma
    if ukloni_sum > 0:
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, ukloni_sum * 3, ukloni_sum * 3, 7, 21)

    # 3. Popravljanje kontrasta
    if popravi_kontrast:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    if crno_belo:
        pil_img = ImageOps.grayscale(pil_img).convert("RGB")

    if ostrina != 1.0:
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(ostrina)

    return pil_img

# Dubinska AI restauracija preko besplatnog Hugging Face GFPGAN modela
def ai_dubinska_restauracija(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name, format="PNG")
        tmp_path = tmp.name

    try:
        # Povezivanje na javni GFPGAN prostor za rekonstrukciju lica i starih slika
        client = Client("Xintao/GFPGAN")
        result_path = client.predict(
            img=handle_file(tmp_path),
            version="v1.4",
            scale=2,
            api_name="/predict"
        )
        # GFPGAN nekad vraća listu [cropped_faces, restored_img] ili putanju
        if isinstance(result_path, (list, tuple)):
            final_path = result_path[-1]
        else:
            final_path = result_path
        return Image.open(final_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- INTERFEJS APLIKACIJE ---
st.title("🖼️ AI Restauracija i Obnova Starih Slika")
st.write("Vratite stari sjaj porodičnim uspomenama – uklonite žutilo, izoštrite lica i obnovite oštećene fotografije.")

prikazi_reklamu()
st.divider()

uploaded_file = st.file_uploader("📂 Izaberite staru fotografiju (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)

    mod = st.radio(
        "Izaberite način obnove:",
        ["✨ Dubinska AI Restauracija Lica (Preporučeno za mutna lica i oštećenja)", "⚡ Brza korekcija boja, kontrasta i žutila"],
        index=0
    )

    restored_image = None

    if "Dubinska AI" in mod:
        st.info("AI model analizira lica i detalje na slici. Obrada traje oko 15–30 sekundi.")
        if st.button("🚀 Pokreni AI Restauraciju", type="primary", use_container_width=True):
            with st.spinner("Veštačka inteligencija rekonstruiše lica i čisti sliku... Sačekajte..."):
                try:
                    # Prvo uklonimo jako žutilo ako ga ima, pa šaljemo AI modelu
                    pred_obrada = brza_restauracija(original_image, 1, True, True, 1.0, False)
                    restored_image = ai_dubinska_restauracija(pred_obrada)
                    st.session_state["restored"] = restored_image
                except Exception as e:
                    st.warning("Besplatni AI server je trenutno preopterećen. Pokrećem naprednu lokalnu obnovu...")
                    restored_image = brza_restauracija(original_image, 3, True, True, 1.8, False)
                    st.session_state["restored"] = restored_image
        if "restored" in st.session_state:
            restored_image = st.session_state["restored"]
    else:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            ukloni_sum = st.slider("Čišćenje šuma i mrlja", 0, 5, 2)
            ostrina = st.slider("Izoštravanje detalja", 0.5, 3.0, 1.6, step=0.1)
        with col_s2:
            neutralisi_zuto = st.checkbox("Ukloni žuti/stari ton sa slike", value=True)
            popravi_kontrast = st.checkbox("Popravi izbledeli kontrast", value=True)
            crno_belo = st.checkbox("Prebaci u čistu crno-belu sliku", value=False)

        restored_image = brza_restauracija(original_image, ukloni_sum, neutralisi_zuto, popravi_kontrast, ostrina, crno_belo)

    if restored_image is not None:
        st.success("✅ Fotografija je obrađena!")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Pre (Original)", use_container_width=True)
        with col2:
            st.image(restored_image, caption="Posle (Obnovljena)", use_container_width=True)

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
