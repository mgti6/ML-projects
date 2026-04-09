import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.database import init_db

st.set_page_config(
    page_title="Cockpit Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ── Sidebar navigation ──────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/48/combo-chart.png", width=40)
st.sidebar.title("Cockpit Manager")
st.sidebar.markdown("---")

pages = {
    "🏠  Accueil": "home",
    "📋  Template & Fonds": "template",
    "⚙️  Générer un cockpit": "generate",
    "🔍  Consulter un cockpit": "view",
    "📥  Import CSSF": "cssf",
    "🗺️  AIF Risk Mapping": "risk_mapping",
    "📤  Export Excel": "export",
}

if "page" not in st.session_state:
    st.session_state.page = "home"

for label, key in pages.items():
    if st.sidebar.button(label, use_container_width=True,
                         type="primary" if st.session_state.page == key else "secondary"):
        st.session_state.page = key
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("v1.0 · SQLite backend")

# ── Page router ─────────────────────────────────────
page = st.session_state.page

if page == "home":
    from pages import home; home.render()
elif page == "template":
    from pages import template; template.render()
elif page == "generate":
    from pages import generate; generate.render()
elif page == "view":
    from pages import view_cockpit; view_cockpit.render()
elif page == "cssf":
    from pages import cssf_import; cssf_import.render()
elif page == "risk_mapping":
    from pages import risk_mapping; risk_mapping.render()
elif page == "export":
    from pages import export; export.render()