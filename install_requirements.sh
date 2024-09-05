# install_requirements.sh

# Update package lists
apt update

# Install system dependencies
apt install -y libgl1-mesa-glx

# Install Python dependencies from requirements.txt
pip install -r requirements.txt
