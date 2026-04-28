import streamlit as st

st.title("About GeoVision")

st.markdown(
    """
    **GeoVision** is a portfolio project demonstrating an end-to-end computer-vision
    + MLOps pipeline for satellite imagery.

    ### Datasets
    - [EuroSAT](https://github.com/phelber/eurosat) — labeled Sentinel-2 patches (13 bands, 10 classes).
    - [Sentinel-2 / ESA Copernicus](https://scihub.copernicus.eu/) — live multi-spectral imagery.

    ### Stack
    PyTorch · TensorFlow · rasterio · geopandas · albumentations · scikit-learn · MLflow ·
    Airflow · Streamlit · folium · Tableau Public · Docker · GitHub Actions.

    ### Source
    [GitHub repository](#)
    """
)
