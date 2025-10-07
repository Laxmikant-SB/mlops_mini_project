mlops_mini_project
==============================
**Emotion Detection — MLOps Project**

This project focuses on building a complete MLOps pipeline for an Emotion Detection model.
Instead of focusing on accuracy improvement, **the main goal was to implement end-to-end ML lifecycle management**, including data versioning, experiment tracking, containerization, CI/CD, and cloud deployment.

**Tech Stack**
Category	          Tools / Libraries
Language	           Python 
Data Versioning	       DVC
Experiment Tracking    MLflow
CI/CD	               GitHub Actions
Containerization	   Docker
Cloud Deployment	   AWS EC2 + ECR
Web Framework	       Flask

**Project Pipeline**
<img width="661" height="693" alt="Screenshot 2025-10-06 234315" src="https://github.com/user-attachments/assets/1848f61c-32c5-49a9-813c-f39662560e70" />


##  Setup & Installation

### Clone the Repository
git clone https://github.com/<your-username>/emotion-detection-mlops.git
cd emotion-detection-mlops

### Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows

### Install Dependencies
pip install -r requirements.txt

### Configure Environment Variables
Copy .env.example to .env and fill in your credentials (e.g. DAGSHUB_PAT, AWS_ACCESS_KEY, etc.)

##  Run Locally
export PYTHONPATH=$(pwd)
python -m flask_app.app
Visit 👉 http://localhost:5000

##  Docker Usage

###  Build Docker Image
docker build -t emotion-detection:latest .

###  Run Docker Container
docker run -p 8080:5000 --env-file .env emotion-detection:latest
Then visit 👉 http://localhost:8080


##  Results
-  Deployed Flask application successfully on AWS EC2
-  Fully automated MLOps pipeline achieved with:
  - DVC for data & model versioning
  - MLflow for experiment tracking
  - GitHub Actions for CI/CD
  - Docker + ECR + EC2 for deployment



