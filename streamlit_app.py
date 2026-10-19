from pathlib import Path
import io
import json
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_mobilenetv2_plantdisease.keras"
CLASS_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = (256, 256)
CONFIDENCE_THRESHOLD = 0.60
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

st.set_page_config(
    page_title="PlantCare | จำแนกโรคจากใบพืช",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { max-width: 760px; padding-top: 1.6rem; padding-bottom: 3rem; }
    .hero { padding: 1.2rem 1.25rem; border: 1px solid rgba(100,140,100,.25); border-radius: 18px; margin-bottom: 1rem; }
    .muted { opacity: .78; font-size: .92rem; }
    div[data-testid="stMetric"] { padding: .7rem; border: 1px solid rgba(128,128,128,.2); border-radius: 12px; }
    .stButton button { width: 100%; border-radius: 12px; min-height: 2.8rem; }
</style>
""", unsafe_allow_html=True)

with open(CLASS_PATH, encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

st.markdown("""
<div class="hero">
  <div style="font-size:2rem">🌿 PlantCare</div>
  <h1 style="margin:.25rem 0 .4rem 0">จำแนกโรคจากใบพืช</h1>
  <p style="margin:0">อัปโหลดภาพใบพืชเพื่อดูผลทำนายจากโมเดล MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="กำลังโหลดโมเดลครั้งแรก…")
def load_model():
    import tensorflow as tf
    loaded = tf.keras.models.load_model(MODEL_PATH, compile=False)
    output_shape = loaded.output_shape
    if not output_shape or output_shape[-1] != len(CLASS_NAMES):
        raise RuntimeError(
            f"โมเดลให้ output {output_shape} แต่ class_names.json มี {len(CLASS_NAMES)} คลาส"
        )
    return loaded


def format_label(class_name: str) -> str:
    plant, disease = class_name.split("___", 1)
    plant = (plant.replace("_(including_sour)", "")
                  .replace("_(maize)", "")
                  .replace(",_bell", " bell")
                  .replace("_", " ").strip())
    disease = disease.replace("_", " ").strip()
    if disease.lower() == "healthy":
        return f"{plant} — ใบปกติ (Healthy)"
    return f"{plant} — {disease}"


def image_suitability_check(rgb_image: Image.Image):
    """Basic color heuristic only; this is NOT a trained leaf detector."""
    sample = rgb_image.copy()
    sample.thumbnail((160, 160))
    arr = np.asarray(sample.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = (maxc - minc) / np.maximum(maxc, 1e-5)
    green = (g > r * 0.88) & (g > b * 0.90) & (g > 0.16) & (g < 0.98)
    yellow_brown = (r > b * 1.12) & (g > b * 1.08) & (r > 0.16) & (g > 0.12) & (saturation > 0.12)
    fraction = float(np.mean(green | yellow_brown))
    return fraction >= 0.015, fraction


def predict(image: Image.Image):
    import tensorflow as tf
    model = load_model()
    likely_plant, fraction = image_suitability_check(image)
    if not likely_plant:
        return {"status": "not_leaf", "plantish_fraction": fraction}

    resized = tf.image.resize(np.asarray(image.convert("RGB")), IMAGE_SIZE, method="bilinear")
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(resized, tf.float32))
    start = time.perf_counter()
    probabilities = np.asarray(model(batch[None, ...], training=False))[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    top_indices = np.argsort(probabilities)[::-1][:3]
    top = [
        {
            "class_name": CLASS_NAMES[int(i)],
            "label": format_label(CLASS_NAMES[int(i)]),
            "confidence": float(probabilities[i]),
        }
        for i in top_indices
    ]
    return {
        "status": "uncertain" if top[0]["confidence"] < CONFIDENCE_THRESHOLD else "ok",
        "top": top,
        "inference_ms": elapsed_ms,
        "plantish_fraction": fraction,
    }

uploaded = st.file_uploader(
    "เลือกรูปใบพืช",
    type=["jpg", "jpeg", "png", "webp"],
    help="รองรับ JPG, PNG และ WEBP ขนาดไม่เกิน 12 MB",
)

if uploaded is not None:
    raw = uploaded.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        st.error("ไฟล์ใหญ่เกิน 12 MB กรุณาเลือกภาพที่เล็กลง")
    else:
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            if image.width < 32 or image.height < 32:
                st.error("ภาพมีขนาดเล็กเกินไป กรุณาเลือกภาพที่ชัดกว่านี้")
            else:
                st.image(image, caption="ภาพที่เลือก", use_container_width=True)
                if st.button("วิเคราะห์ภาพ", type="primary"):
                    try:
                        with st.spinner("กำลังวิเคราะห์ภาพ…"):
                            result = predict(image)
                        st.divider()
                        if result["status"] == "not_leaf":
                            st.warning("ไม่พบลักษณะสีที่คล้ายใบพืชมากพอ จึงหยุดการวิเคราะห์")
                            st.write("ลองใช้ภาพใบพืชที่เห็นชัดและมีแสงเพียงพอ")
                            st.caption("การตรวจนี้เป็นการคัดกรองจากสีแบบง่าย ไม่ใช่โมเดลตรวจจับใบไม้โดยเฉพาะ จึงอาจผิดพลาดได้")
                        else:
                            best = result["top"][0]
                            if result["status"] == "uncertain":
                                st.warning("ผลทำนายยังไม่ชัดเจน ควรลองถ่ายภาพใบเดี่ยวให้ชัดขึ้น")
                            else:
                                st.success("วิเคราะห์เสร็จแล้ว")
                            st.subheader("ผลทำนายอันดับ 1")
                            st.markdown(f"### {best['label']}")
                            st.metric("ความมั่นใจของโมเดล", f"{best['confidence']:.1%}")
                            st.subheader("ผลทำนาย 3 อันดับแรก")
                            for rank, item in enumerate(result["top"], start=1):
                                st.write(f"**{rank}. {item['label']}** — {item['confidence']:.1%}")
                                st.progress(min(max(item["confidence"], 0.0), 1.0))
                            st.caption(f"เวลา inference โดยประมาณ: {result['inference_ms']:.0f} ms")
                            st.info("ผลนี้เป็นการจำแนกจากโมเดล ไม่ใช่การวินิจฉัยยืนยัน และอาจไม่แม่นยำกับภาพถ่ายภาคสนามที่ต่างจากข้อมูลฝึก")
                    except Exception as exc:
                        st.error("เกิดปัญหาระหว่างโหลดโมเดลหรือวิเคราะห์ภาพ")
                        st.caption(f"รายละเอียดสำหรับผู้พัฒนา: {type(exc).__name__}: {exc}")
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("อ่านภาพนี้ไม่ได้ กรุณาเลือกไฟล์ภาพใหม่")

st.divider()
st.caption("โมเดล MobileNetV2 • 38 คลาสของ PlantVillage • ขนาดอินพุต 256×256 RGB")
st.caption("ตัวกรองภาพที่ไม่ใช่ใบพืชเป็น heuristic จากสีเท่านั้น อาจปล่อยภาพอื่นผ่านหรือปฏิเสธใบพืชบางภาพได้")
