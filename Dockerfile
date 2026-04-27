FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by netCDF4 and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libnetcdf-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifacts
COPY get_filled_netcdf.py models.py ./
COPY model/ ./model/
RUN chmod -R a+rX /app

# Output is written to /output — mount a host directory here to retrieve results
RUN mkdir -p /output && chmod 777 /output

ENV MPLCONFIGDIR=/tmp

ENTRYPOINT ["python", "get_filled_netcdf.py"]
CMD ["--help"]
