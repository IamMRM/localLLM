# .devcontainer/Dockerfile
FROM determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.31.1

# Install sudo
USER root
RUN apt-get update && \
    apt-get install -y sudo

# Install Node.js (which includes npm)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && \
    sudo apt-get install -y nodejs


# installing ngrok
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list && \
apt update && apt install ngrok

# Optional: Create a non-root user with sudo privileges
RUN useradd -ms /bin/bash devuser && \
    echo "devuser:devpassword" | chpasswd && \
    adduser devuser sudo

# Switch to the non-root user
USER devuser

WORKDIR /workspace

# Copy the requirements.txt and install Python packages
COPY requirements.txt /workspace/
RUN pip install --upgrade pip && pip install -r requirements.txt