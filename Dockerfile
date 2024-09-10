# Start with the NiftyMIC image as the base
FROM gerardmartijuan/niftymic.multifact AS niftymic

# Install ANTs
FROM antsx/ants:master AS ants

# Final stage
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Copy NiftyMIC and ANTs from previous stages
COPY --from=niftymic /usr/local /usr/local
COPY --from=ants /opt/ants /opt/ants

# Set up environment for ANTs
ENV ANTSPATH="/opt/ants/bin"
ENV PATH="$ANTSPATH:$PATH"

# Copy your application files
COPY app.py /app/
COPY recon /app/recon

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Remove the following line as we've included all necessary packages in requirements.txt
# RUN pip3 install nibabel pandas docker

# Set up X11 forwarding
ENV DISPLAY=:0

# Expose any necessary ports (adjust as needed)
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["python3", "app.py"]