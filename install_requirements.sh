# install_requirements.sh

# Update package lists
sudo apt update

# Install system dependencies
sudo apt install -y libgl1-mesa-glx

# Install Python dependencies from requirements.txt
pip install -r requirements.txt
