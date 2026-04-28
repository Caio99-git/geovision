import streamlit as st

st.set_page_config(
    page_title="GeoVision",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ GeoVision")
st.subheader("Satellite-imagery land-use classification & change detection")

st.markdown(
    """
    Pick a page from the sidebar:

    - **Classify** — upload a Sentinel-2 tile, get a land-use prediction with Grad-CAM.
    - **Change detection** — pick a region and two dates, see what changed.
    - **About** — methodology, datasets, links.

    *Status: scaffold only. Pages will fill in across Phases 2–3.*
    """
)
