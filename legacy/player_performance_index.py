"""Deprecated entrypoint.

The original Colab → Anvil uplink script lived here and contained hardcoded
credentials. It has been replaced by the Streamlit showcase:

    streamlit run app/streamlit_app.py

Core logic now lives under ``src/ppi/``.
"""

raise SystemExit(
    "player_performance_index.py is deprecated. "
    "Run: streamlit run app/streamlit_app.py"
)
