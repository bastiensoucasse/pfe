#!/bin/bash

set -o errexit

install_slicer_from_homebrew() {
	if brew list --cask --versions slicer &>/dev/null; then
		echo "Slicer is already installed."
	else
		echo "Installing Slicer with Homebrew…"
		brew install slicer >/dev/null
		echo "Slicer installed."
	fi
}

install_slicer_on_linux() {
	folder="$(pwd)"
	cd ~

	if [ -d slicer ]; then
		echo "Slicer is already installed at \"$(pwd)/slicer\"."
	else
		echo "Installing Slicer…"
		wget -q -O slicer.tar.gz "https://download.slicer.org/download?os=linux&stability=release"
		tar -xzf slicer.tar.gz
		rm slicer.tar.gz
		mv Slicer-* slicer
		echo "Slicer installed at \"$(pwd)/slicer\"."
	fi

	cd "$folder"
}

install_slicer() {
	if which brew &>/dev/null; then
		install_slicer_from_homebrew
	else
		system=$(uname)

		if [ "$system" != "Darwin" ] && [ "$system" != "Linux" ]; then
			echo "This self-installation script only supports Apple macOS and Linux distributions."
			exit 1
		fi

		if [ "$system" == "Darwin" ]; then
			echo "This self-installation script only supports Homebrew on Apple macOS systems."
			exit 1
		fi

		if [ "$system" == "Linux" ]; then
			install_slicer_on_linux
		fi
	fi
}

check_python() {
	if command -v python &>/dev/null; then
		python_version=$(python --version 2>&1)
		if [[ $python_version == *"3."* ]]; then
			python_interpreter="python"
			pip_interpreter="pip"
		fi
	fi

	if [[ -z $python_interpreter ]] && command -v python3 &>/dev/null; then
		python_version=$(python3 --version 2>&1)
		if [[ $python_version == *"3."* ]]; then
			python_interpreter="python3"
			pip_interpreter="pip3"
		fi
	fi

	if [[ -z $python_interpreter ]]; then
		echo "Error: No suitable Python interpreter found. A Python 3 installation is required."
		exit 1
	fi
}

install_python_requirements() {
	check_python

	echo "Installing the Python requirements…"
	if [[ "$(basename "$(pwd)")" == "requirements" ]]; then
		$pip_interpreter install -q -r "requirements.txt"
	else
		$pip_interpreter install -q -r "requirements/requirements.txt"
	fi
	echo "Python requirements installed."
}

install_dependencies() {
	install_slicer
	install_python_requirements
	echo "Slicer and Python requirements are installed on your system."
}

install_dependencies
