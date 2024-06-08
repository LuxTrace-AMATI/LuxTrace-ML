# Gunakan image python sebagai base image
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy semua file ke container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port aplikasi
EXPOSE 5000

# Jalankan aplikasi
CMD ["python", "main.py"]
