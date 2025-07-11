# Auto-generated stub
# frontend/dashboards/routing.py
import streamlit as st
import requests
import json

st.set_page_config(page_title="QAOA Expert Routing", layout="wide")
st.title("🧠 QAOA Sparse Expert Routing")

model_graph = {"experts": [0, 1, 2, 3]}
tokens = [{"token_id": i} for i in range(8)]

st.code(json.dumps(model_graph, indent=2), language="json")

use_q = st.toggle("Use Quantum (QAOA) Routing", value=True)

if st.button("Run Routing"):
    resp = requests.post("http://localhost:8000/q-routing", json={
        "model_graph": model_graph,
        "token_stream": tokens,
        "use_quantum": use_q
    })
    if resp.ok:
        st.success("Routing Complete")
        st.json(resp.json())
    else:
        st.error("Error occurred")
