#!/bin/bash

# Run the Streamlit tool inside docker, with automated restart

set -x

# Replace with volume path
files="/mnt/volume_ams3_01/files"

# --user to set user so that I can modify/move/delete the docker-created files
docker run -d -p 8501:8501 \
	--restart unless-stopped \
	--user 1000:1000 \
	-v "$files/config":/app/config \
	-v "$files/data":/app/data \
	-v "$files/logs":/app/logs \
	annotation
