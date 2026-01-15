import json
import html
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(page_title="Chatbot Santé — Interface Utilisateur", layout="wide")

# =========================
# Style (plus de couleurs)
# =========================
CUSTOM_CSS = """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

/* Hero */
.hero {
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  border: 1px solid rgba(255,255,255,0.10);
  background: radial-gradient(1200px 400px at 10% 10%, rgba(61, 151, 255, 0.35), rgba(0,0,0,0) 60%),
              radial-gradient(900px 380px at 90% 20%, rgba(0, 209, 178, 0.26), rgba(0,0,0,0) 62%),
              linear-gradient(180deg, rgba(20,24,33,0.95), rgba(20,24,33,0.75));
}
.hero h1 { margin: 0; font-size: 2.0rem; line-height: 1.15; }
.hero p  { margin: 0.5rem 0 0 0; opacity: 0.88; font-size: 1.0rem; }

/* Badge status */
.badges { margin-top: 10px; display:flex; gap:10px; flex-wrap:wrap; }
.badge {
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.90rem;
}
.dot { width:10px; height:10px; border-radius: 999px; display:inline-block; }
.dot-ok { background: #00d1b2; box-shadow: 0 0 0 3px rgba(0, 209, 178, 0.15); }
.dot-bad { background: #ff4d4d; box-shadow: 0 0 0 3px rgba(255, 77, 77, 0.15); }

/* Cards */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.03);
}
.card h3 { margin: 0 0 10px 0; font-size: 1.05rem; }

/* Tags */
.tags { display:flex; gap:8px; flex-wrap:wrap; margin-top: 8px; }
.tag {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(61,151,255,0.25);
  background: rgba(61,151,255,0.10);
  color: rgba(255,255,255,0.92);
  font-size: 0.88rem;
}

/* Secondary tag (for typical symptoms etc.) */
.tag2 {
  border: 1px solid rgba(0,209,178,0.25);
  background: rgba(0,209,178,0.10);
}

/* Small note */
.small-note { font-size: 0.92rem; opacity: 0.85; }

/* Buttons (Streamlit) */
.stButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(61,151,255,0.30) !important;
  background: linear-gradient(180deg, rgba(61,151,255,0.22), rgba(61,151,255,0.10)) !important;
}
.stButton > button:hover {
  border-color: rgba(0,209,178,0.35) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def ping(api_base: str) -> tuple[bool, str]:
    try:
        r = requests.get(f"{api_base}/health", timeout=2)
        if r.status_code == 200:
            return True, "Connecté"
        return False, f"Réponse backend: {r.status_code}"
    except Exception as e:
        return False, f"Indisponible ({type(e).__name__})"


def render_tags(items: list[str], variant: str = "tag"):
    if not items:
        st.write("—")
        return
    safe_items = [html.escape(str(x)) for x in items]
    cls = "tag" if variant == "tag" else "tag tag2"
    tags_html = "<div class='tags'>" + "".join([f"<span class='{cls}'>{x}</span>" for x in safe_items]) + "</div>"
    st.markdown(tags_html, unsafe_allow_html=True)


def post_predict(api_base: str, text: str, top_k: int) -> dict:
    payload = {"text": text, "top_k": top_k}
    r = requests.post(f"{api_base}/predict", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def format_assistant_summary(data: dict) -> str:
    primary = data.get("primary_disease", "—")
    found = data.get("found_symptoms", []) or []
    found_str = ", ".join(found) if found else "—"
    return f"**Résultat :** {primary}\n\n**Symptômes détectés :** {found_str}"


# =========================
# Sidebar (paramètres)
# =========================
st.sidebar.markdown("### Paramètres")
api_url = st.sidebar.text_input("URL du backend (API)", value="http://localhost:8000")
top_k = st.sidebar.slider("Top-K maladies", 1, 5, 3)

col_a, col_b = st.sidebar.columns(2)
with col_a:
    clear = st.button("Vider le chat", use_container_width=True)
with col_b:
    test_conn = st.button("Tester", use_container_width=True)

if clear:
    st.session_state.pop("messages", None)
    st.session_state.pop("last_raw", None)
    st.rerun()

ok, status_msg = ping(api_url)
if test_conn:
    ok, status_msg = ping(api_url)

st.sidebar.markdown("### Statut")
if ok:
    st.sidebar.success(status_msg)
else:
    st.sidebar.error(status_msg)

with st.sidebar.expander("Infos (sécurité)", expanded=False):
    st.caption(
        "Cet assistant fournit une aide indicative basée sur un modèle externe. "
        "En cas de symptômes graves ou urgents, contactez un professionnel de santé."
    )

with st.sidebar.expander("Contrat API attendu", expanded=False):
    st.code(
        json.dumps(
            {
                "GET /health": "HTTP 200",
                "POST /predict body": {"text": "fièvre, toux, fatigue", "top_k": 3},
                "predict response (exemple)": {
                    "primary_disease": "Common Cold",
                    "found_symptoms": ["high_fever", "cough", "fatigue"],
                    "predictions": [
                        {"disease": "Common Cold", "probability": 0.92},
                        {"disease": "Flu", "probability": 0.06},
                        {"disease": "Pneumonia", "probability": 0.02},
                    ],
                    "specialist": "Médecin généraliste",
                    "precautions": ["Repos", "Hydratation", "Consulter si aggravation"],
                    "typical_symptoms": ["cough", "runny_nose", "sneezing"],
                    "disclaimer": "Optionnel",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        language="json",
    )


# =========================
# Header (accueil + assistant santé)
# =========================
dot_class = "dot-ok" if ok else "dot-bad"
st.markdown(
    f"""
<div class="hero">
  <h1>Chatbot Santé — Interface Utilisateur</h1>
  <p>Bonjour ! Je suis votre <b>assistant de santé</b>. Décrivez vos symptômes et je vous proposerai une piste probable ainsi que des conseils.</p>
  <div class="badges">
    <span class="badge"><span class="dot {dot_class}"></span><b>Backend</b> : {html.escape(status_msg)}</span>
    <span class="badge"><b>Mode</b> : Chat + Top-{top_k}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_chat, tab_details = st.tabs(["Chat", "Détails & Debug"])


# =========================
# Tab Chat
# =========================
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Salut ! Décris tes symptômes (ex: « fièvre, toux, fatigue »). "
                    "Je te réponds avec une maladie probable, un spécialiste conseillé et des précautions."
                ),
            }
        ]

    # Affichage historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            details = msg.get("details")
            if details:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1, 1, 1])

                primary = details.get("primary_disease", "—")
                specialist = details.get("specialist", "—")
                confidence = details.get("confidence", None)

                with c1:
                    st.markdown("**Maladie probable**")
                    st.markdown(f"<div style='font-size:1.25rem; font-weight:700;'>{html.escape(primary)}</div>", unsafe_allow_html=True)
                    if confidence is not None:
                        st.caption(f"Confiance (approx.) : {confidence:.2f}")

                with c2:
                    st.markdown("**Symptômes détectés**")
                    render_tags(details.get("found_symptoms", []) or [], variant="tag")

                with c3:
                    st.markdown("**Spécialiste conseillé**")
                    st.markdown(f"<div style='font-size:1.05rem; font-weight:600;'>{html.escape(specialist)}</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("Voir les conseils et le top maladies", expanded=False):
                    precautions = details.get("precautions", []) or []
                    typical = details.get("typical_symptoms", []) or []
                    preds = details.get("predictions", []) or []

                    st.markdown("**Précautions**")
                    if precautions:
                        for p in precautions:
                            st.markdown(f"- {p}")
                    else:
                        st.write("—")

                    st.markdown("**Symptômes fréquents**")
                    if typical:
                        render_tags(typical, variant="tag2")
                    else:
                        st.write("—")

                    if preds:
                        st.markdown("**Top maladies (probabilités)**")
                        df = pd.DataFrame(preds)
                        if {"disease", "probability"}.issubset(df.columns):
                            df = df.set_index("disease")
                            st.bar_chart(df["probability"])
                        else:
                            st.json(preds)

    # Zone de saisie
    user_text = st.chat_input("Tape tes symptômes ici…")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})

        try:
            data = post_predict(api_url, user_text, top_k=top_k)
            st.session_state.last_raw = data

            # Confidence = proba du top-1 si disponible
            confidence = None
            preds = data.get("predictions", []) or []
            if preds and isinstance(preds, list) and isinstance(preds[0], dict) and "probability" in preds[0]:
                try:
                    confidence = float(preds[0]["probability"])
                except Exception:
                    confidence = None

            details = {
                "primary_disease": data.get("primary_disease", "—"),
                "found_symptoms": data.get("found_symptoms", []) or [],
                "predictions": preds,
                "specialist": data.get("specialist", "—"),
                "precautions": data.get("precautions", []) or [],
                "typical_symptoms": data.get("typical_symptoms", []) or [],
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            summary = format_assistant_summary(data)
            st.session_state.messages.append({"role": "assistant", "content": summary, "details": details})
            st.rerun()

        except requests.exceptions.RequestException as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Je n’arrive pas à contacter le service de prédiction.\n\n"
                        f"- URL: `{api_url}`\n"
                        f"- Détail: `{e}`\n\n"
                        "Vérifie que le backend tourne et que `/health` répond."
                    ),
                }
            )
            st.rerun()


# =========================
# Tab Détails & Debug
# =========================
with tab_details:
    st.markdown("#### Dernière réponse brute (debug)")
    raw = st.session_state.get("last_raw")
    if raw:
        st.json(raw)
    else:
        st.info("Aucune réponse enregistrée pour l’instant. Fais une prédiction dans l’onglet Chat.")