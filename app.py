import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

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

# Funkcija za restauraciju slike
def restauriraj_sliku(image, ukloni_sum, popravi_kontrast, ostrina, zasicenje, povecaj_2x):
    # Konverzija iz PIL u OpenCV format (BGR)
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. Uklanjanje šuma i sitnih ogrebotina (Bilateralni filter čuva ivice lica)
    if ukloni_sum > 0:
        d = 5 + (ukloni_sum * 2)
        sigma = ukloni_sum * 15
        img_bgr = cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)

    # 2. Popravljanje izbledelog kontrasta (CLAHE u LAB prostoru boja)
    if popravi_kontrast:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # 3. Povećanje rezolucije (2x Upscale) ako je izabrano
    if povecaj_2x:
        h, w = img_bgr.shape[:2]
        if max(h, w) < 2500:
            img_bgr = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    # Povratak u PIL format za fino podešavanje oštrine i boja
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 4. Izoštravanje detalja
    if ostrina != 1.0:
        enhancer_sharp = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer_sharp.enhance(ostrina)

    # 5. Oživljavanje boja (Zasićenje)
    if zasicenje != 1.0:
        enhancer_color = ImageEnhance.Color(pil_img)
        pil_img = enhancer_color.enhance(zasicenje)

    return pil_img

# --- GLAVNI INTERFEJS APLIKACIJE ---
st.title("🖼️ Restauracija i Obnova Starih Slika")
st.write("Vratite stari sjaj vašim porodičnim uspomenama. Uklonite oštećenja, izoštrite detalje i osvežite izbledele fotografije u par sekundi – potpuno besplatno.")

# Gornja reklama
prikazi_reklamu()

st.divider()

# Učitavanje slike
uploaded_file = st.file_uploader("📂 Izaberite staru fotografiju (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)

    st.subheader("⚙️ Podešavanja obnove")
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        ukloni_sum = st.slider("Čišćenje šuma i ogrebotina", 0, 5, 2, help="Pegla zrnastu teksturu i sitna oštećenja na starom papiru.")
        ostrina = st.slider("Izoštravanje detalja", 0.5, 3.0, 1.6, step=0.1, help="Povećava jasnoću očiju, ivica i tekstura.")
    
    with col_s2:
        zasicenje = st.slider("Oživljavanje boja", 0.0, 2.0, 1.15, step=0.05, help="0.0 = crno-bela slika, 1.0 = originalne boje, >1.0 = življe boje.")
        popravi_kontrast = st.checkbox("Automatski popravi izbledeli kontrast", value=True)
        povecaj_2x = st.checkbox("Povećaj rezoluciju (2x HD Upscale)", value=True)

    # Obrada slike
    with st.spinner("Obnavljam fotografiju... Sačekajte trenutak..."):
        restored_image = restauriraj_sliku(
            original_image, 
            ukloni_sum, 
            popravi_kontrast, 
            ostrina, 
            zasicenje, 
            povecaj_2x
        )

    st.success("✅ Fotografija je uspešno obnovljena!")

    # Prikaz Pre i Posle
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Pre (Original)", use_container_width=True)
    with col2:
        st.image(restored_image, caption="Posle (Obnovljena)", use_container_width=True)

    # Dugme za preuzimanje
    buf = io.BytesIO()
    restored_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="⬇️ Preuzmi obnovljenu sliku (HD PNG)",
        data=byte_im,
        file_name="obnovljena_slika.png",
        mime="image/png",
        use_container_width=True
    )

st.divider()

# Donja reklama
prikazi_reklamu()

# Koristan tekst i Politika privatnosti
with st.expander("ℹ️ Kako funkcioniše obnova slika i Politika privatnosti"):
    st.markdown("""
    **Kako da dobijete najbolje rezultate?**
    * Ako je slika veoma stara i zrnasta, povećajte **Čišćenje šuma** na 3 ili 4.
    * Ako je slika izbledela i požutela, ostavite uključenu opciju **Automatski popravi izbledeli kontrast**.
    * Za crno-bele slike koje želite da ostanu crno-bele bez neželjenih tonova, pomerite **Oživljavanje boja** na `0.0`.
    
    **Privatnost vaših fotografija:**
    Sve fotografije se obrađuju privremeno u memoriji tokom vašeg korišćenja aplikacije i **ne čuvaju se trajno** na našim serverima.
    """)
