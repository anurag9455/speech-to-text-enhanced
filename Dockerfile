# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the app will run
EXPOSE 8501

# Command to run the app when the container starts
CMD ["streamlit", "run", "enhanced_streamlit_speech_to_text.py", "--server.port=8501", "--server.address=0.0.0.0"]
