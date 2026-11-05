"""
streamlit_app.py
----------------
Streamlit Community Cloud default entrypoint alias.
Executes the main platform application in app.py.
"""
import runpy

runpy.run_path("app.py", run_name="__main__")

