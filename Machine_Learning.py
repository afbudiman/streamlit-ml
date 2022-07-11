import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="DATAVIZU",
        # page_icon="👋",
    )

    st.write("# Welcome to ML526! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        ## A Collection of Machine Learning Projects
    """
    )


if __name__ == "__main__":
    run()