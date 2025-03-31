# Hailo-Web


Hailo-Web-App is a web-based application designed for running inference using Hailo AI models. This guide provides step-by-step instructions on setting up and running the application.

## Preparation

Follow these steps to set up the environment and install dependencies:

```sh
# Create a new directory for the project
mkdir Hailo-Web-App

# Navigate to the project directory
cd Hailo-Web-App

# Create a virtual environment with system site packages
python -m venv --system-site-packages env

# Activate the virtual environment
source env/bin/activate

# Clone the repository
git clone https://github.com/KasunThushara/Hailo-Web.git

# Navigate to the cloned repository
cd Hailo-Web

# Install required Python dependencies
pip install -r requirements.txt

# Grant execution permission to the resource download script
chmod +x download_resources.sh

# Run the resource download script
./download_resources.sh
```
## Run Server
Open a new terminal and follow these steps:

```sh
# Navigate to the project directory
cd Hailo-Web-App

# Activate the virtual environment (if not already activated)
source env/bin/activate

# Navigate to the repository
cd Hailo-Web

# Start the server
python3 server.py

```

Once the server is running, open a web browser and enter the following in the address bar:

```sh
http://<pi-ip-address>:5000
```
This will allow you to access the inference interface.

## Project Inspiration
This project is inspired by the [Hailo AI Application Code Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python).





