import json
import html
from datetime import datetime

import pandas as pd
import requests
import gradio as gr


# =========================
# Backend helpers
# =========================
def _norm_base(url: str) -> str:
    return (url or "").strip().rstrip("/")


def ping(api_base: str) -> tuple[bool, str]:
    api_base = _norm_base(api_base)
    if not api_base:
        return False, "URL vide"
    try:
        r = requests.get(f"{api_base}/health", timeout=2)
        if r.status_code == 200:
            return True, "Connecté"
        return False, f"Réponse backend: {r.status_code}"
    except Exception as e:
        return False, f"Indisponible ({type(e).__name__})"


def post_predict(api_base: str, text: str, top_k: int) -> dict:
    api_base = _norm_base(api_base)
    payload = {"text": text, "top_k": int(top_k)}
    r = requests.post(f"{api_base}/predict", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


# =========================
# UI helpers (HTML)
# =========================
def tags_html(items: list[str], variant: str = "tag") -> str:
    if not items:
        return "<span class='muted'>—</span>"
    cls = "tag" if variant == "tag" else "tag tag2"
    safe = [html.escape(str(x)) for x in items]
    return "<div class='tags'>" + "".join([f"<span class='{cls}'>{x}</span>" for x in safe]) + "</div>"


def status_badge(ok: bool, msg: str) -> str:
    dot = "dot-ok" if ok else "dot-bad"
    safe_msg = html.escape(msg)
    return f"""
    <div class="badge">
      <span class="dot {dot}"></span>
      <b>Statut</b> : {safe_msg}
    </div>
    """


def hero_block(ok: bool, msg: str, top_k: int) -> str:
    return f"""
    <div class="hero">
      <div class="hero-title">Chatbot Santé — Interface Utilisateur</div>
      <div class="hero-sub">
        Salut ! Je suis votre <b>assistant de santé</b>.
        Décrivez vos symptômes et je vous proposerai une piste probable ainsi que des conseils.
      </div>
      <div class="badges">
        {status_badge(ok, msg)}
        <div class="badge"><b>Mode</b> : Chat + Top-{int(top_k)}</div>
      </div>
    </div>
    """


def build_details_html(data: dict) -> str:
    primary = html.escape(str(data.get("primary_disease", "—")))
    specialist = html.escape(str(data.get("specialist", "—")))

    found = data.get("found_symptoms", []) or []
    precautions = data.get("precautions", []) or []
    typical = data.get("typical_symptoms", []) or []

    if precautions:
        precautions_html = "<ul>" + "".join(
            [f"<li>{html.escape(str(p))}</li>" for p in precautions]
        ) + "</ul>"
    else:
        precautions_html = "<span class='muted'>—</span>"

    return f"""
    <div class="card">
      <div class="grid">
        <div>
          <div class="k">Maladie probable</div>
          <div class="v big">{primary}</div>
        </div>
        <div>
          <div class="k">Spécialiste conseillé</div>
          <div class="v">{specialist}</div>
        </div>
      </div>

      <div class="section">
        <div class="k">Symptômes détectés</div>
        {tags_html(found, "tag")}
      </div>

      <div class="section">
        <div class="k">Précautions</div>
        {precautions_html}
      </div>

      <div class="section">
        <div class="k">Symptômes fréquents</div>
        {tags_html(typical, "tag2")}
      </div>

      <div class="foot muted">Horodatage: {html.escape(datetime.utcnow().isoformat() + "Z")}</div>
    </div>
    """


def predictions_to_df(data: dict) -> pd.DataFrame:
    preds = data.get("predictions", []) or []
    if not preds:
        return pd.DataFrame(columns=["disease", "probability"])
    df = pd.DataFrame(preds)
    if "disease" not in df.columns:
        df["disease"] = ""
    if "probability" not in df.columns:
        df["probability"] = None
    return df[["disease", "probability"]]


# =========================
# Callbacks
# =========================
def on_test(api_url: str, top_k: int):
    ok, msg = ping(api_url)
    return hero_block(ok, msg, top_k), status_badge(ok, msg)


def on_clear(api_url: str, top_k: int):
    ok, msg = ping(api_url)  # garde un statut cohérent après reset
    return (
        [],                                   # chatbot history
        [],                                   # history_state
        None,                                 # last_raw json
        "",                                   # details html
        pd.DataFrame(columns=["disease", "probability"]),  # table
        hero_block(ok, msg, top_k),
        status_badge(ok, msg),
    )


def on_send(user_text: str, api_url: str, top_k: int, history: list):
    user_text = (user_text or "").strip()
    history = history or []

    if not user_text:
        return "", history, history, None, "", pd.DataFrame(columns=["disease", "probability"])

    # Optimistic UI
    history = history + [(user_text, "…")]

    try:
        data = post_predict(api_url, user_text, top_k)

        primary = data.get("primary_disease", "—")
        found = data.get("found_symptoms", []) or []
        found_str = ", ".join(found) if found else "—"

        bot_msg = f"**Résultat :** {primary}\n\n**Symptômes détectés :** {found_str}"
        history[-1] = (user_text, bot_msg)

        details = build_details_html(data)
        df = predictions_to_df(data)

        return "", history, history, data, details, df

    except requests.exceptions.RequestException as e:
        err = (
            "Je n’arrive pas à contacter le service de prédiction.\n\n"
            f"- URL: `{api_url}`\n"
            f"- Détail: `{e}`\n\n"
            "Vérifie que le backend tourne et que `/health` répond."
        )
        history[-1] = (user_text, err)
        return "", history, history, None, "", pd.DataFrame(columns=["disease", "probability"])


# =========================
# CSS + Theme
# =========================
CSS = """
.hero{
  border-radius:18px; padding:18px 18px 14px 18px;
  border:1px solid rgba(255,255,255,.10);
  background:
    radial-gradient(1200px 400px at 10% 10%, rgba(61,151,255,.35), rgba(0,0,0,0) 60%),
    radial-gradient(900px 380px at 90% 20%, rgba(0,209,178,.26), rgba(0,0,0,0) 62%),
    linear-gradient(180deg, rgba(20,24,33,.95), rgba(20,24,33,.75));
}
.hero-title{ font-size:2.0rem; font-weight:800; line-height:1.15; }
.hero-sub{ margin-top:8px; opacity:.90; font-size:1.02rem; }
.badges{ margin-top:12px; display:flex; gap:10px; flex-wrap:wrap; }

.badge{
  display:inline-flex; align-items:center; gap:8px;
  padding:6px 10px; border-radius:999px;
  border:1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.04);
  font-size:.92rem;
}
.dot{ width:10px; height:10px; border-radius:999px; display:inline-block; }
.dot-ok{ background:#00d1b2; box-shadow:0 0 0 3px rgba(0,209,178,.15); }
.dot-bad{ background:#ff4d4d; box-shadow:0 0 0 3px rgba(255,77,77,.15); }

.card{
  border:1px solid rgba(255,255,255,.10);
  border-radius:16px; padding:14px 14px;
  background: rgba(255,255,255,.03);
}
.grid{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
.k{ font-size:.92rem; opacity:.85; margin-bottom:6px; }
.v{ font-size:1.02rem; font-weight:600; }
.big{ font-size:1.25rem; font-weight:800; }

.section{ margin-top:14px; }
.tags{ display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.tag{
  display:inline-block; padding:6px 10px; border-radius:999px;
  border:1px solid rgba(61,151,255,.25);
  background: rgba(61,151,255,.10);
  color: rgba(255,255,255,.92);
  font-size:.88rem;
}
.tag2{
  border:1px solid rgba(0,209,178,.25);
  background: rgba(0,209,178,.10);
}
.muted{ opacity:.75; }
.foot{ margin-top:10px; font-size:.85rem; }
"""

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="teal")


# =========================
# App
# =========================
with gr.Blocks(title="Chatbot Santé") as demo:
    history_state = gr.State([])

    hero = gr.HTML(hero_block(False, "Non testé", 3))
    gr.Markdown("Outil pédagogique : ne remplace pas un avis médical.", elem_classes=["muted"])

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### Paramètres")
            api_url = gr.Textbox(label="URL du backend (API)", value="http://localhost:8000")
            top_k = gr.Slider(1, 5, value=3, step=1, label="Top-K maladies")

            with gr.Row():
                btn_test = gr.Button("Tester connexion")
                btn_clear = gr.Button("Vider le chat", variant="stop")

            gr.Markdown("### Statut")
            status_box = gr.HTML(status_badge(False, "Non testé"))

            with gr.Accordion("Contrat API attendu", open=False):
                gr.Code(
                    json.dumps(
                        {
                            "GET /health": "HTTP 200",
                            "POST /predict body": {"text": "fièvre, toux, fatigue", "top_k": 3},
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    language="json",
                    label="",
                )

        with gr.Column(scale=2, min_width=520):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(height=440, label="")

            user_text = gr.Textbox(label="", placeholder="Ex: fièvre, toux, fatigue")
            btn_send = gr.Button("Envoyer", variant="primary")

            with gr.Accordion("Détails & Debug", open=False):
                details = gr.HTML("")
                preds_table = gr.Dataframe(
                    value=pd.DataFrame(columns=["disease", "probability"]),
                    label="Top maladies (probabilités)",
                    interactive=False,
                    wrap=True,
                )
                last_raw = gr.JSON(label="Dernière réponse brute (JSON)", value=None)

    # Events
    btn_test.click(on_test, inputs=[api_url, top_k], outputs=[hero, status_box])

    btn_clear.click(
        on_clear,
        inputs=[api_url, top_k],
        outputs=[chatbot, history_state, last_raw, details, preds_table, hero, status_box],
    )

    btn_send.click(
        on_send,
        inputs=[user_text, api_url, top_k, history_state],
        outputs=[user_text, chatbot, history_state, last_raw, details, preds_table],
    )

    user_text.submit(
        on_send,
        inputs=[user_text, api_url, top_k, history_state],
        outputs=[user_text, chatbot, history_state, last_raw, details, preds_table],
    )

# Gradio 6: css + theme à passer dans launch()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=theme)