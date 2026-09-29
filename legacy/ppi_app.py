"""Deprecated entrypoint.

The original Anvil + ElephantSQL team predictor lived here and contained
hardcoded database credentials. Team ranking is now part of the Streamlit app
(`Squad ranking` tab) using local CSV data.

    streamlit run app/streamlit_app.py
"""

raise SystemExit(
    "ppi_app.py is deprecated. "
    "Run: streamlit run app/streamlit_app.py"
)
