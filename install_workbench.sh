#!/bin/bash

set -e

echo "[INFO] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y libglu1-mesa libxrender1 libxt6 libxext6 mesa-utils wget unzip

echo "[INFO] Downloading Workbench..."
wget -q https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip
unzip -q workbench-linux64-v1.5.0.zip
sudo mv workbench /usr/local/workbench
sudo chmod +x /usr/local/workbench/bin_linux64/wb_command

echo "[INFO] Workbench installed at /usr/local/workbench"
echo "[INFO] You may want to add the following to your ~/.bashrc:"
# Add to PATH (temporary)
export PATH=$PATH:/usr/local/workbench/bin_linux64

# Optional: make this change permanent by editing ~/.bashrc
echo 'export PATH=$PATH:/usr/local/workbench/bin_linux64' >> ~/.bashrc


