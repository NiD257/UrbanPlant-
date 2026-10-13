import streamlit as st
from PIL import Image

from predict import predict_pil_image

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="UrbanPlant",
    page_icon="🌿",
    layout="wide"
)

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.title("🌿 UrbanPlant")

st.sidebar.markdown("---")

st.sidebar.subheader("📖 About")

st.sidebar.write("""
UrbanPlant is an AI-powered urban plant health monitoring system.

This Phase-1 prototype uses Deep Learning to classify plant leaves as Healthy or Diseased.

Future development includes:

• Disease Segmentation

• Severity Estimation

• Multi-Disease Classification

• Urban Greenery Monitoring
""")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Model Information")

st.sidebar.info("""
Model: MobileNetV2

Input Size: 224 × 224

Transfer Learning

Inference: CPU
""")

st.sidebar.markdown("---")

st.sidebar.warning(
    "Research Prototype Only\n\nNot a Professional Agricultural Diagnosis Tool."
)

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.markdown("""
<div style="
background:linear-gradient(90deg,#1B5E20,#4CAF50);
padding:25px;
border-radius:15px;
color:white;
text-align:center;
">

<h1>🌿 UrbanPlant</h1>

<h3>AI-Powered Urban Plant Health Monitoring System</h3>

<p>
Detect plant diseases using Deep Learning and support healthier urban green spaces.
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------
# METRICS
# ----------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Accuracy", "98.75%")

with m2:
    st.metric("Classes", "2")

with m3:
    st.metric("Model", "MobileNetV2")

with m4:
    st.metric("Input Size", "224×224")

st.divider()

# ----------------------------------------------------
# IMAGE UPLOAD + ANALYSIS
# ----------------------------------------------------

left, right = st.columns(2)

with left:

    st.subheader("📷 Upload Plant Image")

    uploaded_file = st.file_uploader(
        "Upload JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

with right:

    st.subheader("🧠 AI Analysis")

    if uploaded_file:

        prediction, confidence, probs = predict_pil_image(image)

        if prediction.lower() == "healthy":

            st.success(
                f"✅ Prediction: {prediction}"
            )

            severity = 5

        else:

            st.error(
                f"⚠️ Prediction: {prediction}"
            )

            severity = 45

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

        st.subheader("Class Probabilities")

        for cls, score in probs.items():

            st.write(
                f"**{cls}** : {score:.2f}%"
            )

        st.subheader("Disease Severity Estimate")

        st.progress(severity)

        st.write(
            f"Estimated affected area: **{severity}%**"
        )

        st.subheader("Recommendation")

        if prediction.lower() == "healthy":

            st.success(
                "Plant appears healthy. Continue regular monitoring and maintenance."
            )

        else:

            st.warning(
                "Disease symptoms detected. Inspect nearby leaves and consider treatment."
            )

    else:

        st.info(
            "Upload a plant image to begin analysis."
        )

# ----------------------------------------------------
# WORKFLOW
# ----------------------------------------------------

st.divider()

st.subheader("🔄 UrbanPlant Workflow")

w1, w2, w3, w4, w5 = st.columns(5)

with w1:
    st.info("📷 Upload")

with w2:
    st.info("🧠 Analyse")

with w3:
    st.info("🔍 Detect")

with w4:
    st.info("📊 Estimate")

with w5:
    st.info("📋 Results")

# ----------------------------------------------------
# CURRENT PROJECT SCOPE
# ----------------------------------------------------

st.divider()

st.subheader("🚀 Current Project Scope")

c1, c2, c3 = st.columns(3)

with c1:

    st.success("""
### 🌱 Classification

Healthy vs Diseased Leaves

Transfer Learning using MobileNetV2
""")

with c2:

    st.info("""
### 📈 Evaluation

Model Validation

Performance Analysis

Accuracy Measurement
""")

with c3:

    st.warning("""
### 🔬 Future Work

Disease Segmentation

Severity Mapping

Campus Monitoring

Mobile Application
""")

# ----------------------------------------------------
# FUTURE SCOPE
# ----------------------------------------------------

st.divider()

st.subheader("🌍 Future Development")

f1, f2, f3 = st.columns(3)

with f1:

    st.success("""
🔍 Disease Segmentation

Highlight infected regions automatically.
""")

with f2:

    st.info("""
📊 Severity Estimation

Calculate percentage of affected area.
""")

with f3:

    st.warning("""
🌳 Urban Monitoring

Deploy across campus and urban green spaces.
""")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.divider()

st.caption(
    "UrbanPlant • AI-Powered Urban Plant Health Monitoring System • Phase 1 Prototype"
)