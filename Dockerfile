# Use an official Python image as a base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install required Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]