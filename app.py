import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Cattle Breed Detector & Classifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #2a5298;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .breed-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .sdg-card {
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Lazy import inference pipeline to prevent slow initial page load
@st.cache_resource
def get_pipeline():
    from src.pipeline import CattleInferencePipeline
    return CattleInferencePipeline()

pipeline = get_pipeline()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/cow.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🎯 Breed Detection & Classifier", "🌐 UN SDGs Alignment", "📐 Architecture & Method"])

# st.sidebar.markdown("---")
# st.sidebar.subheader("Project Authors")
# st.sidebar.write("**Atharv Sutrave** (Roll No. 34108)")
# st.sidebar.write("**Atharva Chudappa** (Roll No. 34117)")
# st.sidebar.caption("College Mini Project — AI & Deep Learning")

# ---------------------------------------------------------
# PAGE 1: BREED DETECTION & CLASSIFICATION
# ---------------------------------------------------------
if page == "🎯 Breed Detection & Classifier":
    st.markdown('<div class="main-header">AI-Based Cattle Breed Detection & Profiling</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Two-Stage Deep Learning System: YOLOv8 Cattle Detection + EfficientNet Breed Classification</div>', unsafe_allow_html=True)

    col_input, col_display = st.columns([1, 2])

    with col_input:
        st.subheader("1. Select Input Image")
        input_source = st.radio("Choose Image Source:", ["Sample Image", "Upload Image"], horizontal=True)

        selected_image = None

        if input_source == "Sample Image":
            sample_breeds = ['Gir', 'Sahiwal', 'Holstein_Friesian', 'Kankrej', 'Murrah', 'Ongole', 'Hallikar', 'Jersey', 'Tharparkar', 'Red_Sindhi']
            chosen_breed = st.selectbox("Select Cattle Breed Sample:", sample_breeds)
            
            breed_folder = os.path.join(".", chosen_breed)
            if os.path.exists(breed_folder):
                available_files = [f for f in os.listdir(breed_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if available_files:
                    sample_file = available_files[0]
                    selected_image = Image.open(os.path.join(breed_folder, sample_file)).convert("RGB")
                    st.info(f"Loaded sample image from **{chosen_breed}** dataset.")

        else:
            uploaded_file = st.file_uploader("Upload a Cattle Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")

    with col_display:
        if selected_image is not None:
            st.subheader("2. AI Analysis & Inference")
            with st.spinner("Executing 2-Stage YOLO Detection & Breed Classification..."):
                results = pipeline.process_image(selected_image)

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.write("**Stage 1: YOLO Detection Bounding Box**")
                st.image(results["annotated_image"], use_container_width=True)
            with img_col2:
                st.write("**Cropped Cattle ROI (Classifier Input)**")
                st.image(results["cropped_image"], use_container_width=True)

            pred = results["prediction"]
            info = results["breed_info"]

            st.markdown("---")
            st.subheader("3. Classification Output")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown(f'<div class="breed-badge">Predicted Breed: {pred["predicted_breed"]}</div>', unsafe_allow_html=True)
                st.write("")
                st.metric(label="Prediction Confidence", value=f"{pred['confidence']}%")
                st.caption(f"Status: **{pred['status']}**")

            with res_col2:
                st.write("**Top Class Probabilities**")
                top_preds = pred["top_predictions"]
                breeds = list(top_preds.keys())
                probs = [v * 100 for v in top_preds.values()]

                fig, ax = plt.subplots(figsize=(5, 2.5))
                y_pos = np.arange(len(breeds))
                ax.barh(y_pos, probs, color='#2a5298')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(breeds, fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel('Probability (%)', fontsize=8)
                ax.set_xlim(0, 100)
                plt.tight_layout()
                st.pyplot(fig)

            # Breed Knowledge Profile
            st.markdown("---")
            st.subheader(f"📋 Breed Feature Profile — {info.get('name', pred['predicted_breed'])}")

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown(f"**🌍 Native Region / Origin:** {info.get('native_region', 'N/A')}")
                st.markdown(f"**🐄 Primary Purpose:** {info.get('primary_purpose', 'N/A')}")
                st.markdown(f"**🥛 Average Milk Yield:** {info.get('milk_yield', 'N/A')}")
                st.markdown(f"**☀️ Heat Tolerance / Climate:** {info.get('heat_tolerance', 'N/A')}")

            with p_col2:
                st.markdown(f"**🎨 Physical Traits:** {info.get('physical_traits', 'N/A')}")
                st.markdown(f"**⚖️ Average Adult Weight:** {info.get('average_weight', 'N/A')}")
                st.markdown(f"**🛡️ Disease Resistance:** {info.get('disease_resistance', 'N/A')}")
                st.markdown(f"**⭐ Special Characteristics:** {info.get('special_characteristics', 'N/A')}")

        else:
            st.info("👈 Please select a sample image or upload a cattle photo to view detection and breed identification results.")

# ---------------------------------------------------------
# PAGE 2: UN SDGS ALIGNMENT
# ---------------------------------------------------------
elif page == "🌐 UN SDGs Alignment":
    st.markdown('<div class="main-header">Alignment with UN Sustainable Development Goals (SDGs)</div>', unsafe_allow_html=True)
    st.write("Beyond technical computer vision metrics, this project connects AI innovation to the United Nations 2030 Agenda for Sustainable Development.")

    sdg_data = [
        {"sdg": "SDG 2: Zero Hunger", "color": "#DDA63A", "desc": "Supports sustainable food and dairy production through accurate indigenous breed documentation and breed-appropriate livestock management."},
        {"sdg": "SDG 8: Decent Work & Economic Growth", "color": "#A21942", "desc": "Provides farmers and livestock workers with an accessible decision-support tool to improve productivity, breeding outcomes, and rural livelihoods."},
        {"sdg": "SDG 9: Industry, Innovation & Infrastructure", "color": "#FD6925", "desc": "Applies state-of-the-art deep learning (YOLO localization + Transfer Learning classification) as digital infrastructure for precision agriculture."},
        {"sdg": "SDG 12: Responsible Consumption & Production", "color": "#BF8B2E", "desc": "Encourages breed-appropriate feed, water, and heat management matching breed traits, reducing resource wastage in tropical livestock farming."},
        {"sdg": "SDG 13: Climate Action", "color": "#3F7E44", "desc": "Highlights climate adaptability and heat tolerance attributes to identify heat-resilient indigenous breeds suited for changing weather patterns."},
        {"sdg": "SDG 15: Life on Land", "color": "#56C02B", "desc": "Promotes conservation and digital record-keeping of indigenous zebu bovine breeds to prevent genetic erosion and preserve biodiversity."}
    ]

    for item in sdg_data:
        st.markdown(f"""
            <div style="background-color: {item['color']}; padding: 1.2rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: white;">{item['sdg']}</h3>
                <p style="margin-top: 0.5rem; font-size: 1.05rem;">{item['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 3: ARCHITECTURE & METHOD
# ---------------------------------------------------------
elif page == "📐 Architecture & Method":
    st.markdown('<div class="main-header">System Architecture & Methodology</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔄 Two-Stage Deep Learning Pipeline

    1. **Stage 1: YOLO Object Detection**
       - Model: Ultralytics YOLOv8
       - Role: Detects and localizes cattle in the input image, eliminating background noise (vegetation, humans, buildings).
       - Output: Crops the precise Region of Interest (ROI) surrounding the animal.

    2. **Stage 2: EfficientNet / MobileNet Transfer Learning Classifier**
       - Model: Fine-tuned Transfer Learning model initialized with ImageNet weights.
       - Inputs: Cropped cattle ROI (resized to 224x224).
       - Augmentations: Random horizontal flip, small rotation, color jitter, and ImageNet standard normalization.
       - Output: Class probabilities via Softmax activation.

    3. **Confidence Thresholding**
       - If top class probability is below confidence threshold (e.g. 45%), system categorizes result as *"Unknown / Low Confidence"* to prevent forced unreliable predictions.

    4. **Breed Knowledge Base Integration**
       - Retrieves structured profile data (Native Region, Purpose, Milk Yield, Heat Tolerance, Traits) from `breeds_info.json`.
    """)
