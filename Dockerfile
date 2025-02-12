# Use an official Python image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional Jupyter components
RUN pip install --no-cache-dir ipykernel psycopg2-binary

# Expose the Jupyter Notebook port
EXPOSE 8888

# Set the default working directory inside Jupyter
CMD ["jupyter-lab", "--LabApp.notebook_dir=/app", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--LabApp.token=''"]