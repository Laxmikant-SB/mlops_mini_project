FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy Flask app files
COPY flask_app/ /app/

# Copy model files
COPY model/vectorizer.pkl /app/model/vectorizer.pkl
COPY model/model.pkl /app/model/model.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -m nltk.downloader stopwords wordnet

# Expose port
EXPOSE 5000

# Start the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
