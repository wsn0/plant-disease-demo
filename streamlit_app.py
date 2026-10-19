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
CONFIDENCE_THRESHOLD = 0.90
MIN_CONFIDENCE_MARGIN = 0.25
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

st.set_page_config(
    page_title="PlantCare | จำแนกโรคจากใบพืช",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { max-width: 800px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .hero { padding: 1.35rem 1.4rem; border: 1px solid rgba(100,140,100,.28); border-radius: 20px; margin-bottom: 1.1rem; background: rgba(100,140,100,.045); }
    .eyebrow { font-size: .82rem; letter-spacing: .04em; text-transform: uppercase; opacity: .72; }
    .muted { opacity: .78; font-size: .92rem; }
    .result-primary { padding: 1rem 1.1rem; border: 1px solid rgba(70,145,85,.45); border-radius: 16px; background: rgba(70,145,85,.07); }
    .result-rank { font-size: .78rem; opacity: .75; margin-bottom: .25rem; }
    .result-secondary { padding: .8rem .9rem; border: 1px solid rgba(128,128,128,.22); border-radius: 14px; min-height: 100%; }
    .result-secondary .label { font-size: .92rem; line-height: 1.4; font-weight: 600; }
    .result-secondary .score { font-size: 1.05rem; font-weight: 650; margin-top: .45rem; }
    .stButton button { width: 100%; border-radius: 12px; min-height: 2.8rem; }
</style>
""", unsafe_allow_html=True)

with open(CLASS_PATH, encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

st.markdown("""
<div class="hero">
  <div class="eyebrow">🌿 PlantCare</div>
  <h1 style="margin:.25rem 0 .5rem 0">ตรวจสอบสุขภาพใบพืช</h1>
  <p style="margin:0">อัปโหลดภาพใบพืชที่ต้องการตรวจสอบ ระบบจะประเมินว่าภาพใกล้เคียงกับกลุ่มโรคที่รองรับหรือไม่</p>
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
    confidence = top[0]["confidence"]
    margin = confidence - top[1]["confidence"] if len(top) > 1 else confidence
    # Conservative screening, not a true out-of-distribution detector.
    accepted = confidence >= CONFIDENCE_THRESHOLD and margin >= MIN_CONFIDENCE_MARGIN
    return {
        "status": "ok" if accepted else "unsupported",
        "top": top,
        "inference_ms": elapsed_ms,
        "plantish_fraction": fraction,
        "confidence_margin": margin,
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
                st.caption("เพื่อผลลัพธ์ที่ดี ควรใช้ภาพใบพืชที่ชัดเจน มีแสงเพียงพอ และให้ใบพืชเป็นส่วนหลักของภาพ")
                if st.button("ตรวจสอบภาพ", type="primary"):
                    try:
                        with st.spinner("กำลังวิเคราะห์ภาพ…"):
                            result = predict(image)
                        st.divider()
                        if result["status"] in ("not_leaf", "unsupported"):
                            st.warning("ยังไม่สามารถประเมินภาพนี้ได้อย่างน่าเชื่อถือ")
                            st.write("ภาพอาจไม่ใช่ใบพืชที่ระบบรองรับ หรือมีลักษณะแตกต่างจากตัวอย่างที่ระบบรู้จัก")
                            st.write("ลองอัปโหลดภาพใบพืชที่เห็นชัด โดยให้ใบพืชเป็นส่วนหลักของภาพ และหลีกเลี่ยงฉากหลังหรือวัตถุอื่นที่รบกวน")
                            st.caption("ระบบคัดกรองจากสีและรูปแบบคะแนนที่ได้ ซึ่งอาจปฏิเสธภาพใบพืชจริงหรือยังยอมรับภาพที่ไม่เหมาะสมได้")
                        else:
                            best = result["top"][0]
                            st.success("ประเมินภาพเสร็จแล้ว")
                            with st.container(border=True):
                                st.markdown('<div class="result-rank">อันดับ 1 · ผลที่ระบบประเมินว่าน่าจะใกล้เคียงที่สุด</div>', unsafe_allow_html=True)
                                st.markdown(f"### {best['label']}")
                                st.markdown(f"**คะแนนความมั่นใจของโมเดล: {best['confidence']:.1%}**")
                                st.progress(min(max(best["confidence"], 0.0), 1.0))

                            st.markdown("#### ผลลัพธ์อื่นที่เป็นไปได้")
                            secondary_cols = st.columns(2)
                            for col, rank, item in zip(secondary_cols, (2, 3), result["top"][1:3]):
                                with col:
                                    with st.container(border=True):
                                        st.markdown(f'<div class="result-rank">อันดับ {rank}</div>', unsafe_allow_html=True)
                                        st.markdown(f'<div class="label">{item["label"]}</div>', unsafe_allow_html=True)
                                        st.markdown(f'<div class="score">{item["confidence"]:.1%}</div>', unsafe_allow_html=True)
                                        st.progress(min(max(item["confidence"], 0.0), 1.0))

                            st.info("ข้อควรทราบ: เปอร์เซ็นต์นี้เป็นคะแนนความมั่นใจที่คำนวณโดยระบบ ไม่ใช่ความน่าจะเป็นที่ยืนยันว่าโรคดังกล่าวถูกต้อง และไม่ใช่การวินิจฉัยจากผู้เชี่ยวชาญ ผลอาจคลาดเคลื่อนได้ โดยเฉพาะภาพที่ต่างจากตัวอย่างที่ระบบรองรับ")
                            st.caption(f"ใช้เวลาประมวลผลประมาณ {result['inference_ms']:.0f} มิลลิวินาที")
                    except Exception as exc:
                        st.error("เกิดปัญหาระหว่างโหลดโมเดลหรือวิเคราะห์ภาพ")
                        st.caption(f"รายละเอียดสำหรับผู้พัฒนา: {type(exc).__name__}: {exc}")
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("อ่านภาพนี้ไม่ได้ กรุณาเลือกไฟล์ภาพใหม่")

st.divider()
st.caption("เครื่องมือนี้ใช้ช่วยคัดกรองเบื้องต้นเท่านั้น ไม่ควรใช้แทนการตรวจสอบจากผู้เชี่ยวชาญหรือใช้ตัดสินใจเรื่องการรักษาพืชเพียงอย่างเดียว")
