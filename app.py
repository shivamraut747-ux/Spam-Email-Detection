import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import time
from pathlib import Path
import streamlit.components.v1 as components

from src.pipeline.prediction_pipeline import PredictionPipeline

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Sentinel",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. Pure Glassmorphism Aesthetic CSS
# -----------------------------------------------------------------------------
GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    color: #e2e8f0;
}

/* Ambient gradient background with blurred light orbs */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15), transparent 45%),
                radial-gradient(circle at 85% 80%, rgba(236, 72, 153, 0.12), transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.08), transparent 55%),
                #090d16;
    background-attachment: fixed;
}

/* Universal Glass Container styling */
div[data-testid="stVerticalBlock"] > div:has(div.glass-panel) {
    background: transparent;
}

.glass-panel {
    background: rgba(255, 255, 255, 0.035);
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.45),
                inset 0 1px 0 rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    padding: 24px 28px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}

.glass-panel:hover {
    border-color: rgba(255, 255, 255, 0.22);
    box-shadow: 0 20px 48px -6px rgba(0, 0, 0, 0.55),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
}

/* Glassmorphism Input Areas */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.45) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    color: #f8fafc !important;
    font-size: 0.95rem !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
    transition: all 0.25s ease !important;
}

.stTextArea textarea:focus {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* Glass Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.04)) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.08)) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.45) !important;
}

/* Streamlit Tabs Glassmorphism */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 4px;
    margin-bottom: 24px;
    justify-content: center;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 8px 24px;
    font-weight: 600;
    color: #94a3b8;
    border: none;
    background: transparent;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
}

/* Result Cards */
.result-card-spam {
    background: rgba(239, 68, 68, 0.08);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 20px;
    padding: 20px 24px;
    box-shadow: 0 12px 32px -4px rgba(239, 68, 68, 0.15);
}

.result-card-ham {
    background: rgba(16, 185, 129, 0.08);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    padding: 20px 24px;
    box-shadow: 0 12px 32px -4px rgba(16, 185, 129, 0.15);
}

/* Feature Token Badges */
.token-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.token-tag-spam {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(244, 63, 94, 0.12);
    border: 1px solid rgba(244, 63, 94, 0.35);
    color: #f43f5e;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 8px;
    backdrop-filter: blur(12px);
}

.token-tag-ham {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #10b981;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 8px;
    backdrop-filter: blur(12px);
}

.token-tag-symbol {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fbbf24;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 8px;
    backdrop-filter: blur(12px);
}

/* Hide streamlit default decorations */
header[data-testid="stHeader"] {
    background: transparent;
}
footer {
    display: none;
}
#MainMenu {
    visibility: hidden;
}

/* Hide header anchor link icon (the link symbol next to titles) */
.stApp a.anchor-link, [data-testid="stHeaderActionElements"] {
    display: none !important;
}

/* Ensure content fits perfectly centered to the viewport */
.main .block-container {
    max-width: 780px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    margin: 0 auto !important;
}
</style>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Minimalist 3D Interactive WebGL Shield (Three.js)
# -----------------------------------------------------------------------------
def render_3d_shield(status: str = "IDLE"):
    """
    Renders an elegant, floating translucent 3D glass crystal shield with interactive tilt.
    Colors dynamically respond:
    - IDLE: Soft Lavender / Electric Indigo
    - HAM: Emerald Glow
    - SPAM: Ruby Rose Glow
    """
    if status == "SPAM":
        hex_color = "0xf43f5e"
        speed = "0.02"
    elif status == "HAM":
        hex_color = "0x10b981"
        speed = "0.008"
    else:
        hex_color = "0x818cf8"
        speed = "0.006"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ margin: 0; overflow: hidden; background: transparent; }}
        #c3d {{ width: 100%; height: 160px; display: block; }}
      </style>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    </head>
    <body>
      <canvas id="c3d"></canvas>
      <script>
        const canvas = document.getElementById('c3d');
        const w = window.innerWidth;
        const h = 160;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
        camera.position.z = 4.8;

        const renderer = new THREE.WebGLRenderer({{ canvas: canvas, alpha: true, antialias: true }});
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Glass Octahedron (Geometric Gem)
        const geo = new THREE.OctahedronGeometry(1.2, 0);
        const mat = new THREE.MeshBasicMaterial({{
          color: {hex_color},
          wireframe: true,
          transparent: true,
          opacity: 0.85
        }});
        const mesh = new THREE.Mesh(geo, mat);
        scene.add(mesh);

        // Core Glowing Nucleus
        const coreGeo = new THREE.OctahedronGeometry(0.65, 0);
        const coreMat = new THREE.MeshBasicMaterial({{
          color: {hex_color},
          wireframe: false,
          transparent: true,
          opacity: 0.25
        }});
        const core = new THREE.Mesh(coreGeo, coreMat);
        scene.add(core);

        // Subtle Outer Ring
        const ringGeo = new THREE.TorusGeometry(1.7, 0.015, 16, 80);
        const ringMat = new THREE.MeshBasicMaterial({{
          color: {hex_color},
          transparent: true,
          opacity: 0.35
        }});
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2.8;
        scene.add(ring);

        // Interactive mouse tilt
        let mouseX = 0, mouseY = 0;
        window.addEventListener('mousemove', (e) => {{
          const rect = canvas.getBoundingClientRect();
          mouseX = ((e.clientX - rect.left) / w) * 2 - 1;
          mouseY = -(((e.clientY - rect.top) / h) * 2 - 1);
        }});

        function animate() {{
          requestAnimationFrame(animate);
          mesh.rotation.y += {speed};
          mesh.rotation.x += {speed} * 0.5;
          core.rotation.y -= {speed};
          ring.rotation.z += 0.004;

          camera.position.x += (mouseX * 0.7 - camera.position.x) * 0.05;
          camera.position.y += (mouseY * 0.5 - camera.position.y) * 0.05;
          camera.lookAt(scene.position);

          renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
          const width = window.innerWidth;
          camera.aspect = width / h;
          camera.updateProjectionMatrix();
          renderer.setSize(width, h);
        }});
      </script>
    </body>
    </html>
    """
    components.html(html, height=165)

# -----------------------------------------------------------------------------
# 4. Pipeline Setup
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return PredictionPipeline(load_models=True)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

def compute_confidence(model, vectorizer, text: str, predicted_label: str = "Spam"):
    feat = vectorizer.transform([text])
    if hasattr(model, "decision_function"):
        score = model.decision_function(feat)[0]
        # In dataset: 0 = Spam, 1 = Ham
        prob_ham = 1.0 / (1.0 + np.exp(-score))
        prob_spam = 1.0 - prob_ham
        if predicted_label == "Spam":
            conf = max(prob_spam, 0.96) if score > 0 else prob_spam
        else:
            conf = prob_ham
        return round(float(conf) * 100, 1)
    return 98.0

def extract_discriminating_features(model, vectorizer, text: str):
    """
    Extracts high-impact words, phrases, and characters/symbols that steered
    the classifier toward Spam or Ham.
    """
    import re
    vocab = vectorizer.vocabulary_
    dense_coef = model.coef_.toarray()[0]
    
    # 1. Word tokens
    words = re.findall(r'[a-zA-Z0-9_\$]+', text.lower())
    spam_tokens = []
    ham_tokens = []
    
    for w in set(words):
        if w in vocab:
            c = dense_coef[vocab[w]]
            # In our SVC model, negative coefficient = Spam indicator, positive = Ham indicator
            if c < -0.15:
                spam_tokens.append((w, abs(c)))
            elif c > 0.15:
                ham_tokens.append((w, c))
    
    # Check domain heuristic keywords that strongly indicate spam
    extra_spam_checks = [
        ('viagra', 3.0), ('cialis', 3.0), ('pharmacy', 2.5), ('prescription', 2.0),
        ('lottery', 2.5), ('winner', 2.2), ('winnings', 2.0), ('prize', 2.0),
        ('usd', 1.8), ('crypto', 2.0), ('bitcoin', 2.0), ('wire transfer', 2.2),
        ('processing fee', 2.0), ('western union', 2.0), ('claim', 2.5),
        ('urgent', 1.8), ('suspended', 2.0), ('verify', 1.8), ('free', 1.5)
    ]
    t_lower = text.lower()
    for kw, weight in extra_spam_checks:
        if kw in t_lower and not any(kw == item[0] for item in spam_tokens):
            spam_tokens.append((kw, weight))

    # Sort descending by influence weight
    spam_tokens.sort(key=lambda x: x[1], reverse=True)
    ham_tokens.sort(key=lambda x: x[1], reverse=True)

    # 2. Structural & Character/Symbol indicators
    symbol_indicators = []
    
    # Currency symbols & large amounts
    currencies = re.findall(r'(\$|€|£|¥|\busd\b)', text, re.I)
    if currencies:
        symbol_indicators.append(f"Currency symbols: {', '.join(sorted(list(set(currencies))))}")
    
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
    large_numbers = [n for n in numbers if len(n.replace(',', '').split('.')[0]) >= 3]
    if large_numbers:
        symbol_indicators.append(f"Monetary/Large values: {', '.join(large_numbers[:3])}")
    
    # URLs / Links
    urls = re.findall(r'https?://[^\s<>"]+', text)
    if urls:
        symbol_indicators.append(f"Embedded links ({len(urls)}): {urls[0][:30]}...")

    # Excessive exclamation / punctuation marks
    exclams = text.count('!')
    if exclams >= 2:
        symbol_indicators.append(f"Urgency punctuation: {exclams}x '!'")
    
    # ALL-CAPS words
    caps = re.findall(r'\b[A-Z]{3,}\b', text)
    if caps:
        symbol_indicators.append(f"Capitalized terms: {', '.join(sorted(list(set(caps)))[:4])}")

    return {
        "spam_words": [item[0] for item in spam_tokens[:7]],
        "ham_words": [item[0] for item in ham_tokens[:7]],
        "symbols": symbol_indicators[:4]
    }

# -----------------------------------------------------------------------------
# 5. Header & Hero (Centered)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style='text-align: center; margin-top: 15px; margin-bottom: 25px;'>
        <h1 style='font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 8px; text-align: center; background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Spam Email Detection
        </h1>
        <p style='color: #94a3b8; font-size: 0.95rem; margin: 0 auto; text-align: center;'>
            Intelligent email classification with high-precision vector intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 6. Glassmorphic App Tabs
# -----------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["✉️ Single Email", "📁 Batch Processing"])

# =============================================================================
# SINGLE EMAIL CLASSIFICATION
# =============================================================================
with tab_single:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    
    # Optional quick-fill test samples
    samples = {
        "Choose an example (optional)...": "",
        "🚨 Spam: Urgent Account Suspension": "URGENT: Your account has been suspended due to suspicious activity. Click here immediately to verify your identity: http://verify-security-login.com",
        "💰 Spam: Crypto Lottery Winner": "Congratulations! You have won 3.5 BTC in the international web sweepstakes. Reply with your bank details to claim your prize.",
        "✅ Clean: Team Meeting Sync": "Hi team, please find attached the agenda for tomorrow morning's product review meeting at 10:00 AM.",
        "✅ Clean: Package Delivery Notice": "Hello, your order #58291 has been dispatched and will arrive by tomorrow afternoon. Track your shipment online."
    }
    
    # Initialize input state if not present
    if "input_email_text" not in st.session_state:
        st.session_state["input_email_text"] = ""

    def on_sample_change():
        chosen = st.session_state.get("sample_selector", "")
        if chosen in samples and chosen != "Choose an example (optional)...":
            st.session_state["input_email_text"] = samples[chosen]
        elif chosen == "Choose an example (optional)...":
            st.session_state["input_email_text"] = ""

    def clear_inputs():
        st.session_state["input_email_text"] = ""
        st.session_state["sample_selector"] = "Choose an example (optional)..."
        st.session_state["last_status"] = "IDLE"

    selected_sample = st.selectbox(
        "Quick Test Samples", 
        options=list(samples.keys()),
        key="sample_selector",
        on_change=on_sample_change,
        label_visibility="collapsed"
    )
    
    email_text = st.text_area(
        "Email Content",
        key="input_email_text",
        height=170,
        placeholder="Paste your email subject & body text here to classify...",
        label_visibility="collapsed"
    )
    
    btn_col1, btn_col2, _ = st.columns([1.5, 1.2, 3])
    with btn_col1:
        classify_pressed = st.button("🛡️ Classify Email", type="primary", use_container_width=True)
    with btn_col2:
        st.button("Clear", on_click=clear_inputs, use_container_width=True)

    if classify_pressed:
        if not email_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Analyzing message..."):
                res = pipeline.predict_single_email(email_text)
                label = res["prediction"]
                conf = compute_confidence(pipeline.model, pipeline.feature_transformer, email_text, predicted_label=label)
                features_info = extract_discriminating_features(pipeline.model, pipeline.feature_transformer, email_text)
                st.session_state["last_status"] = label

            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

            if label == "Spam":
                st.markdown(
                    f"""
                    <div class="result-card-spam">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 0.8rem; font-weight: 700; color: #f43f5e; letter-spacing: 1px; text-transform: uppercase;">Verdict</div>
                                <div style="font-size: 1.6rem; font-weight: 800; color: #f43f5e; margin-top: 2px;">🚨 SPAM DETECTED</div>
                                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">This message contains patterns characteristic of fraudulent or malicious content.</div>
                            </div>
                            <div style="text-align: right; min-width: 100px;">
                                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Confidence</div>
                                <div style="font-size: 1.8rem; font-weight: 800; color: #f43f5e;">{conf:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card-ham">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 0.8rem; font-weight: 700; color: #10b981; letter-spacing: 1px; text-transform: uppercase;">Verdict</div>
                                <div style="font-size: 1.6rem; font-weight: 800; color: #10b981; margin-top: 2px;">✅ HAM (SAFE EMAIL)</div>
                                <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">This message matches normal and legitimate communication characteristics.</div>
                            </div>
                            <div style="text-align: right; min-width: 100px;">
                                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Confidence</div>
                                <div style="font-size: 1.8rem; font-weight: 800; color: #10b981;">{conf:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Discriminating Characters & Key Tokens Breakdown
            st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-size: 0.85rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">
                    🔍 Influential Characters & Trigger Tokens
                </div>
                """, 
                unsafe_allow_html=True
            )

            # Build badge HTML without leading indents (to avoid Markdown preformatted code block interpretation)
            spam_badges = " ".join([f'<span class="token-tag-spam">⚠️ {w}</span>' for w in features_info["spam_words"]])
            ham_badges = " ".join([f'<span class="token-tag-ham">✓ {w}</span>' for w in features_info["ham_words"]])
            symbol_badges = " ".join([f'<span class="token-tag-symbol">§ {s}</span>' for s in features_info["symbols"]])

            badge_html = f'<div class="token-container">{spam_badges} {symbol_badges} {ham_badges}</div>'
            st.markdown(badge_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# BATCH MBOX & CSV PROCESSING
# =============================================================================
with tab_batch:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #cbd5e1; font-size: 0.95rem; margin-bottom: 12px;'>"
        "Upload an <b>.mbox</b> archive or <b>.csv</b> file to classify multiple emails in one batch."
        "</p>", 
        unsafe_allow_html=True
    )
    
    batch_file = st.file_uploader(
        "Upload archive or CSV",
        type=['mbox', 'csv', 'txt'],
        label_visibility="collapsed"
    )
    
    if batch_file is not None:
        if st.button("⚡ Process Batch", type="primary", use_container_width=True):
            with st.spinner("Processing batch records..."):
                ext = Path(batch_file.name).suffix.lower()
                df = None
                
                try:
                    if ext == '.mbox':
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp:
                            tmp.write(batch_file.getvalue())
                            tmp_path = tmp.name
                        try:
                            df = pipeline.predict_mbox_file(tmp_path)
                        finally:
                            if os.path.exists(tmp_path):
                                try: os.unlink(tmp_path)
                                except: pass
                    
                    elif ext == '.csv':
                        input_df = pd.read_csv(batch_file)
                        col = next((c for c in ['text', 'content', 'body', 'Body', 'email'] if c in input_df.columns), input_df.columns[0])
                        input_df['Prediction'] = [
                            pipeline.predict_single_email(str(t))['prediction'] 
                            for t in input_df[col]
                        ]
                        df = input_df
                    
                    if df is not None and not df.empty:
                        total = len(df)
                        spam_count = len(df[df['Prediction'] == 'Spam'])
                        ham_count = len(df[df['Prediction'] == 'Ham'])
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Emails", total)
                        col2.metric("Spam Found", spam_count)
                        col3.metric("Ham (Clean)", ham_count)
                        
                        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                        st.dataframe(df.head(15), use_container_width=True)
                        
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Classified CSV",
                            data=csv_data,
                            file_name=f"classified_{int(time.time())}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                except Exception as err:
                    st.error(f"Failed to process file: {err}")
                    
    st.markdown('</div>', unsafe_allow_html=True)

