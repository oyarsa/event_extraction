#!/bin/sh
# Set up python dependencies using either uv or pip-tools, with OS-specific requirements

if [ -z "${VIRTUAL_ENV}" ]; then
	echo "ERROR: No active Python virtual environment detected."
	echo "Please activate a virtual environment and run this script again."
	exit 1
fi

if [ "$(uname)" = "Linux" ]; then
	requirements_file="requirements-linux.txt"
elif [ "$(uname)" = "Darwin" ]; then
	requirements_file="requirements-mac.txt"
else
	echo "ERROR: Only Linux (CUDA) and macOS are supported."
	exit 1
fi

# Use uv or pip-tools to manage dependencies.
if command -v uv >/dev/null 2>&1; then
	compile_cmd="uv pip compile"
	sync_cmd="uv pip sync"
elif command -v pip-compile >/dev/null 2>&1; then
	compile_cmd="pip-compile"
	sync_cmd="pip-sync"
else
	echo "Neither uv nor pip-tools are installed. Install either one and try again."
	exit 1
fi

# Compile requirements if the file doesn't exist, then sync
if [ ! -f "$requirements_file" ]; then
	echo "Creating new requirements file $requirements_file using $compile_cmd."
	$compile_cmd requirements.in -o $requirements_file
fi

echo "Syncing dependencies using $sync_cmd."
$sync_cmd $requirements_file
