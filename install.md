```markdown
# Detailed Installation Guide

## Hardware Requirements

### Minimum Requirements
- CPU: 4+ cores
- RAM: 16GB
- Storage: 10GB free space
- GPU: NVIDIA GPU with 8GB VRAM (for GPU acceleration)

### Recommended Requirements
- CPU: 8+ cores
- RAM: 32GB
- Storage: 20GB free space
- GPU: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, A5000, A6000)

## Software Requirements

### Windows
1. Install Python 3.8 or later
2. Install Visual Studio Build Tools 2019 or later
3. Install CUDA Toolkit 11.8 or later (for GPU support)
4. Install Git

### Linux (Ubuntu/Debian)
1. Update system packages:
```bash
sudo apt update
sudo apt upgrade
```
Install required packages:

```bash
sudo apt install python3-dev python3-pip git build-essential
```
### Install CUDA (for GPU support):

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install cuda-11-8
```

### MacOS

Install Xcode Command Line Tools:
```bash
xcode-select --install
```
Install Homebrew (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Install Python:
```bash
brew install python@3.10
```
### Installation Steps
#### Method 1: Using pip

Clone the repository:
```bash
clone https://github.com/deepseek-ai/Janus.git
cd Janus
```
### Create and activate virtual environment:

Windows:
```bash
python -m venv janus_env
janus_env\Scripts\activate
```

Linux/MacOS:
```bash
python -m venv janus_env
source janus_env/bin/activate
```
Install dependencies:

```bash
pip install -e .
```
#### Method 2: Using conda

Create and activate conda environment:

```bash
conda create -n janus python=3.10
conda activate janus
```

Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install Janus:
```bash
pip install -e .
```
### Optional Features

Install Gradio demo dependencies:

```bash
pip install -e .[gradio]
```
Install development tools:

```bash
pip install -e .[dev]
```
Install linting tools:

```bash
pip install -e .[lint]
```
