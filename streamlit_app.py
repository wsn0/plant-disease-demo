from pathlib import Path
import html
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
OOD_PATH = BASE_DIR / "ood_stats.npz"
# เกณฑ์ความมั่นใจของ softmax ใช้แยกภาพนอกชุดข้อมูลไม่ได้ (รูปสลัดยังได้ ~88%)
# จึงใช้เป็นแค่ "ธงเตือน" ส่วนการตัดสินว่าภาพแปลกไหมใช้ระยะ Mahalanobis (ood_stats.npz) แทน
LOW_CONFIDENCE = 0.50
# ใส่ตัวเลขเพื่อ override เกณฑ์ใน ood_stats.npz (None = ใช้ค่าในไฟล์)
OOD_WARN_OVERRIDE = None
OOD_REJECT_OVERRIDE = None
MIN_CONFIDENCE_MARGIN = 0.15
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

st.set_page_config(
    page_title="PlantCare | จำแนกโรคจากใบพืช",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background: #f7faf6; color: #203326; }
    [data-testid="stHeader"] { background: rgba(247,250,246,.92); }
    .block-container { max-width: 820px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .hero { padding: 1.35rem 1.4rem; border: 1px solid #dce9d9; border-radius: 20px; margin-bottom: 1.1rem; background: #ffffff; }
    .hero h1 { font-size: 1.9rem; line-height: 1.3; }
    .hero p { color: #3c5242; line-height: 1.6; }
    .eyebrow { font-size: .82rem; letter-spacing: .04em; color: #527557; }
    .muted { color: #607064; font-size: .92rem; }
    .result-primary { padding: 1.1rem 1.25rem 1.2rem; border: 1px solid #c8dfc5; border-radius: 18px; background: #ffffff; margin: .4rem 0 1.2rem; }
    .result-rank { font-size: .82rem; color: #607064; margin-bottom: .5rem; }
    .result-plant { font-size: 1.7rem; font-weight: 700; line-height: 1.3; margin: 0; color: #203326; }
    .result-disease { font-size: 1.2rem; font-weight: 600; margin: .15rem 0 .9rem; color: #365f3c; }
    .conf-row { display: flex; justify-content: space-between; font-size: .92rem; color: #304735; margin-bottom: .35rem; }
    .conf-row b { font-weight: 700; }
    .bar { height: 8px; background: #e6eee3; border-radius: 99px; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: #47794d; border-radius: 99px; }
    .section-title { font-size: 1.05rem; font-weight: 650; margin: .2rem 0 .6rem; color: #203326; }
    .result-secondary { background: #ffffff; border: 1px solid #e0e8dd; border-radius: 14px; padding: .85rem 1rem 1rem; height: 100%; }
    .result-secondary .label { font-size: .95rem; line-height: 1.5; font-weight: 600; color: #304735; }
    .result-secondary .score { font-size: .9rem; margin-top: .5rem; color: #527557; }
    .result-secondary .bar { height: 6px; margin-top: .45rem; }
    [data-testid="stImage"] img { max-height: 420px; object-fit: contain; border-radius: 14px; background: #eef3ec; }
    .stButton { margin: .4rem 0 .2rem; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    .stButton button { width: 100%; border-radius: 12px; min-height: 2.8rem; background: #47794d; border-color: #47794d; }
    .stButton button:hover { background: #365f3c; border-color: #365f3c; }
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


@st.cache_resource(show_spinner=False)
def load_ood():
    """โหลดตัวตรวจภาพนอกชุดข้อมูล (OOD) + โมเดลที่ให้ทั้ง feature และ probability ในรอบเดียว"""
    import tensorflow as tf
    model = load_model()
    gap = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    dual = tf.keras.Model(model.input, [gap.output, model.output])
    if not OOD_PATH.exists():
        return {"dual": dual, "stats": None}
    z = np.load(OOD_PATH)
    stats = {
        "mu": z["mu"].astype(np.float64),
        "prec": z["prec"].astype(np.float64),
        "warn": OOD_WARN_OVERRIDE if OOD_WARN_OVERRIDE is not None else float(z["warn_threshold"]),
        "reject": OOD_REJECT_OVERRIDE if OOD_REJECT_OVERRIDE is not None else float(z["reject_threshold"]),
    }
    return {"dual": dual, "stats": stats}


def mahalanobis_score(feature: np.ndarray, stats: dict) -> float:
    """ระยะถึงคลาสที่ใกล้ที่สุด ยิ่งสูงยิ่งต่างจากภาพที่โมเดลเคยเห็น"""
    diff = stats["mu"] - feature[None, :]                 # (K, 1280)
    return float(np.min(np.einsum("kj,jl,kl->k", diff, stats["prec"], diff)))


def quality_warnings(rgb_image: Image.Image):
    """ตรวจคุณภาพภาพแบบเบา ๆ — เตือนเท่านั้น ไม่บล็อก"""
    g = np.asarray(rgb_image.convert("L").resize((256, 256)), dtype=np.float32)
    notes = []
    if g.mean() < 55:
        notes.append("ภาพค่อนข้างมืด")
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    if lap.var() < 25:
        notes.append("ภาพค่อนข้างเบลอ")
    return notes


PLANT_NAMES_TH = {
    "Apple": "แอปเปิล",
    "Blueberry": "บลูเบอร์รี",
    "Cherry_(including_sour)": "เชอร์รี",
    "Cherry": "เชอร์รี",
    "Corn_(maize)": "ข้าวโพด",
    "Corn": "ข้าวโพด",
    "Grape": "องุ่น",
    "Orange": "ส้ม",
    "Peach": "พีช",
    "Pepper,_bell": "พริกหวาน",
    "Pepper": "พริก",
    "Potato": "มันฝรั่ง",
    "Raspberry": "ราสป์เบอร์รี",
    "Soybean": "ถั่วเหลือง",
    "Squash": "ฟักทอง",
    "Strawberry": "สตรอว์เบอร์รี",
    "Tomato": "มะเขือเทศ",
}

DISEASE_NAMES_TH = {
    "healthy": "ใบปกติ",
    "apple scab": "โรคแคงเกอร์/ตกสะเก็ดแอปเปิล",
    "black rot": "โรคผลเน่าดำ",
    "cedar apple rust": "โรคราสนิมซีดาร์-แอปเปิล",
    "powdery mildew": "โรคราแป้ง",
    "cercospora leaf spot gray leaf spot": "โรคใบจุดเซอร์โคสปอรา",
    "common rust": "โรคราสนิมทั่วไป",
    "northern leaf blight": "โรคใบไหม้ทางเหนือ",
    "esca (black measles)": "โรคเอสกา",
    "leaf blight (isariopsis leaf spot)": "โรคใบไหม้",
    "haunglongbing (citrus greening)": "โรคกรีนนิงของส้ม",
    "bacterial spot": "โรคใบจุดแบคทีเรีย",
    "early blight": "โรคใบไหม้ระยะแรก",
    "late blight": "โรคใบไหม้ระยะท้าย",
    "leaf scorch": "โรคใบไหม้แห้ง",
    "leaf mold": "โรคราใบ",
    "septoria leaf spot": "โรคใบจุดเซพโทเรีย",
    "spider mites two-spotted spider mite": "ความเสียหายจากไรแมงมุม",
    "target spot": "โรคใบจุดเป้า",
    "tomato yellow leaf curl virus": "โรคใบหงิกเหลืองจากไวรัส",
    "tomato mosaic virus": "โรคโมเสกมะเขือเทศ",
    "two-spotted spider mite": "ความเสียหายจากไรแมงมุม",
    "bacterial spot": "โรคใบจุดแบคทีเรีย",
    "leaf spot": "โรคใบจุด",
    "mosaic virus": "โรคโมเสกจากไวรัส",
}

def format_label(class_name: str) -> dict:
    """Return human-readable Thai names while preserving the original class label."""
    if "___" not in class_name:
        return {"plant": class_name, "disease": "ไม่ทราบคลาส", "original": class_name}

    plant_key, disease_key = class_name.split("___", 1)
    plant_lookup = plant_key.replace("_(including_sour)", "").replace("_(maize)", "")
    plant_lookup = plant_lookup.replace(",_bell", "").replace("_", " ").strip()
    plant_name = PLANT_NAMES_TH.get(plant_key, PLANT_NAMES_TH.get(plant_lookup, plant_lookup))

    disease_original = disease_key.replace("_", " ").strip()
    disease_lookup = disease_original.lower()
    disease_name = DISEASE_NAMES_TH.get(disease_lookup)
    if disease_name is None:
        # Make any unmapped label readable rather than exposing dataset formatting.
        disease_name = disease_original.replace("(", "(").strip()
        disease_name = disease_name[:1].upper() + disease_name[1:]

    return {
        "plant": plant_name,
        "disease": disease_name,
        "original": f"{plant_lookup} — {disease_original}",
        "healthy": disease_lookup == "healthy",
    }


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
    ood = load_ood()
    likely_plant, fraction = image_suitability_check(image)
    quality = quality_warnings(image)
    if not likely_plant:
        return {"status": "not_leaf", "plantish_fraction": fraction, "quality": quality}

    resized = tf.image.resize(np.asarray(image.convert("RGB")), IMAGE_SIZE, method="bilinear")
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(resized, tf.float32))
    start = time.perf_counter()
    feat, probs = ood["dual"](batch[None, ...], training=False)
    probabilities = np.asarray(probs)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    top_indices = np.argsort(probabilities)[::-1][:3]
    top = [
        {
            "class_name": CLASS_NAMES[int(i)],
            "display": format_label(CLASS_NAMES[int(i)]),
            "confidence": float(probabilities[i]),
        }
        for i in top_indices
    ]
    confidence = top[0]["confidence"]
    margin = confidence - top[1]["confidence"] if len(top) > 1 else confidence

    stats = ood["stats"]
    ood_score = mahalanobis_score(np.asarray(feat)[0].astype(np.float64), stats) if stats else None

    flags = []          # เหตุผลที่ควรระวังผลลัพธ์
    if stats is None:
        status = "ok" if (confidence >= 0.90 and margin >= 0.25) else "unsupported"
    elif ood_score > stats["reject"]:
        status = "unsupported"                      # ต่างจากข้อมูลที่เคยเห็นมาก → ไม่ทาย
    else:
        status = "ok"
        if ood_score > stats["warn"]:
            status = "caution"
            flags.append("ภาพมีลักษณะต่างจากภาพที่ใช้ฝึกโมเดลพอสมควร (เช่น ฉากหลัง แสง หรือมุมถ่าย)")
        if confidence < LOW_CONFIDENCE:
            status = "caution"
            flags.append("ความมั่นใจของโมเดลค่อนข้างต่ำ")
        elif margin < MIN_CONFIDENCE_MARGIN:
            status = "caution"
            flags.append("ผลอันดับ 1 และ 2 ใกล้กันมาก")
    if quality and status == "ok":
        status = "caution"
    return {
        "status": status,
        "top": top,
        "flags": flags + quality,
        "inference_ms": elapsed_ms,
        "plantish_fraction": fraction,
        "confidence_margin": margin,
        "ood_score": ood_score,
        "ood_warn": stats["warn"] if stats else None,
        "ood_reject": stats["reject"] if stats else None,
    }

def _disease_line(display: dict) -> str:
    return "🌿 สถานะ: ใบปกติ" if display.get("healthy", False) else f"🦠 โรคที่คาดการณ์: {display['disease']}"


def _bar(confidence: float) -> str:
    pct = min(max(confidence, 0.0), 1.0) * 100
    return f'<div class="bar"><span style="width:{pct:.1f}%"></span></div>'


def render_primary(item: dict):
    d = item["display"]
    st.markdown(
        f'''<div class="result-primary">
  <div class="result-rank">อันดับ 1 · ผลที่ระบบประเมินว่าใกล้เคียงที่สุด</div>
  <p class="result-plant">🌱 พืช: {html.escape(d["plant"])}</p>
  <p class="result-disease">{html.escape(_disease_line(d))}</p>
  <div class="conf-row"><span>ความมั่นใจในการทำนาย</span><b>{item["confidence"]:.1%}</b></div>
  {_bar(item["confidence"])}
</div>''',
        unsafe_allow_html=True,
    )


def render_secondary(rank: int, item: dict):
    d = item["display"]
    status = "🌿 ใบปกติ" if d.get("healthy", False) else f"🦠 {d['disease']}"
    st.markdown(
        f'''<div class="result-secondary">
  <div class="result-rank">อันดับ {rank}</div>
  <div class="label">🌱 {html.escape(d["plant"])}<br>{html.escape(status)}</div>
  <div class="score">ความมั่นใจในการทำนาย: {item["confidence"]:.1%}</div>
  {_bar(item["confidence"])}
</div>''',
        unsafe_allow_html=True,
    )


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
                            st.markdown(
                                "- ภาพอาจไม่ใช่ใบพืชที่ระบบรองรับ หรือมีลักษณะแตกต่างจากตัวอย่างที่ระบบรู้จัก\n"
                                "- ลองอัปโหลดภาพใบพืชที่เห็นชัด โดยให้ใบพืชเป็นส่วนหลักของภาพ และหลีกเลี่ยงฉากหลังหรือวัตถุอื่นที่รบกวน"
                            )
                            st.caption("ระบบคัดกรองจากสีและรูปแบบคะแนนที่ได้ ซึ่งอาจปฏิเสธภาพใบพืชจริงหรือยังยอมรับภาพที่ไม่เหมาะสมได้")
                        else:
                            if result["status"] == "caution":
                                st.warning("ประเมินได้ แต่ควรระวังผลลัพธ์นี้: " + " · ".join(result["flags"]))
                            else:
                                st.success("ประเมินภาพเสร็จแล้ว")
                            render_primary(result["top"][0])
                            st.markdown('<div class="section-title">ผลลัพธ์ทางเลือก</div>', unsafe_allow_html=True)
                            for col, rank, item in zip(st.columns(2, gap="medium"), (2, 3), result["top"][1:3]):
                                with col:
                                    render_secondary(rank, item)
                            st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
                            st.warning("โปรดทราบ: คะแนนนี้เป็นค่าการทำนายจากระบบ ไม่ใช่การยืนยันว่าพืชเป็นโรคดังกล่าวจริง หากอาการไม่ตรงกับภาพหรือยังไม่แน่ใจ ควรตรวจสอบข้อมูลเพิ่มเติมหรือปรึกษาผู้เชี่ยวชาญ")
                        if st.query_params.get("debug") == "1" and result.get("ood_score") is not None:
                            st.caption(f"[debug] OOD {result['ood_score']:.0f} · warn>{result['ood_warn']:.0f} · reject>{result['ood_reject']:.0f} · {result['status']}")
                    except Exception as exc:
                        st.error("เกิดปัญหาระหว่างโหลดโมเดลหรือวิเคราะห์ภาพ")
                        st.caption(f"รายละเอียดสำหรับผู้พัฒนา: {type(exc).__name__}: {exc}")
        except (UnidentifiedImageError, OSError, ValueError):
            st.error("อ่านภาพนี้ไม่ได้ กรุณาเลือกไฟล์ภาพใหม่")

st.divider()
st.caption("PlantCare เป็นเครื่องมือช่วยประเมินเบื้องต้น ผลอาจผิดพลาดได้ ควรใช้ร่วมกับการสังเกตอาการจริงและคำแนะนำจากผู้เชี่ยวชาญ")
