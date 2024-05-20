#!/bin/bash

# Run the Streamlit tool directly. Debug only.

set -x

export ANNOTATION_CONFIG_PATH=deployment/config.yaml  
streamlit run src/annotation/Start_Page.py
