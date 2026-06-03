import streamlit as st
import pickle, numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recogniser · SVM",
    page_icon="✍",
    layout="centered",
)

# ── Load model (cached so it loads only once) ────────────────────────
@st.cache_resource
def load_model():
    with open("svm_digit_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Preprocessing ─────────────────────────────────────────────────────
def preprocess(pil_img):
    '''
    Canvas image (RGBA) → (1, 784) float array for the SVM.
    Pipeline: grayscale → tight crop → 20% padding → 28×28 → normalise → flatten
    '''
    gray    = pil_img.convert("L")
    bbox    = gray.getbbox()
    if bbox is None:
        return None                         # blank canvas
    cropped = gray.crop(bbox)
    w, h    = cropped.size
    pad     = max(int(max(w, h) * 0.20), 4)
    padded  = ImageOps.expand(cropped, border=pad, fill=0)
    resized = padded.resize((28, 28), Image.Resampling.LANCZOS)
    arr     = np.array(resized, dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)

# ── Confidence helper ─────────────────────────────────────────────────
def get_confidence(vec, digit):
    classes = getattr(model, "classes_", list(range(10)))
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(vec)[0]
            pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
            return dict(pairs).get(digit, max(proba)), pairs[:3]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            s     = model.decision_function(vec)[0]
            e     = np.exp(s - np.max(s))
            proba = e / e.sum()
            pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
            return dict(pairs).get(digit, max(proba)), pairs[:3]
        except Exception:
            pass
    return 1.0, [(digit, 1.0)]

# ── UI ────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#7C3AED'>✍ Digit Recogniser</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#94A3B8'>SVM · scikit-learn · MNIST 28×28</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

col_canvas, col_result = st.columns([1.2, 1])

with col_canvas:
    st.markdown("#### 🖊 Draw a digit (0–9)")
    canvas_result = st_canvas(
        fill_color   = "rgba(255,255,255,1)",
        stroke_width = 28,
        stroke_color = "#FFFFFF",
        background_color = "#000000",
        width  = 400,
        height = 400,
        drawing_mode = "freedraw",
        key = "canvas",
    )

with col_result:
    st.markdown("#### 🎯 Prediction")

    if canvas_result.image_data is not None:
        # canvas returns RGBA numpy array
        rgba = canvas_result.image_data.astype(np.uint8)
        pil  = Image.fromarray(rgba, mode="RGBA")
        vec  = preprocess(pil)

        if vec is not None:
            digit          = int(model.predict(vec)[0])
            confidence, t3 = get_confidence(vec, digit)
            pct            = confidence * 100
            bar_color      = "green" if pct >= 70 else "orange"

            # Big digit
            st.markdown(
                f"<div style='text-align:center;font-size:120px;"
                f"font-weight:bold;color:#7C3AED;line-height:1'>{digit}</div>",
                unsafe_allow_html=True,
            )

            # Confidence bar
            st.markdown(f"**Confidence: {pct:.1f}%**")
            st.progress(int(pct))

            # Top-3
            st.markdown("**Top-3 Candidates**")
            medals = ["🥇", "🥈", "🥉"]
            for i, (cls, score) in enumerate(t3):
                c1, c2 = st.columns([2, 1])
                c1.write(f"{medals[i]} Digit **{cls}**")
                c2.write(f"{score*100:.1f}%")
        else:
            st.markdown(
                "<div style='text-align:center;font-size:80px;color:#334155'>?</div>",
                unsafe_allow_html=True,
            )
            st.info("Draw a digit on the canvas to see the prediction.")
    else:
        st.info("Draw a digit on the canvas to see the prediction.")

st.markdown("---")
st.caption(
    f"Model: **{type(model).__name__}**  |  "
    f"Input: 28×28 = 784 features  |  Classes: 0–9"
)