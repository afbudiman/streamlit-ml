import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="ML256",
        # page_icon="👋",
    )

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.write("# Welcome to ML526! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        ## A Collection of Machine Learning Projects
    """
    )


if __name__ == "__main__":
    run()