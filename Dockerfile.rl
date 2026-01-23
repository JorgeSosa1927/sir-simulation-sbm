
FROM python:3.10-slim

# Install system dependencies
# MuJoCo needs GL libraries
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-dev \
    libgl1 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
# Downgrade build tools to allow installation of older gym versions (Malformed version string issue)
RUN pip install --no-cache-dir "pip==21.3.1" "setuptools==65.5.0" "wheel==0.38.4"

# Note: minari[all] includes many datasets dependencies.
# We explicitly install gymnasium[mujoco] as requested.
RUN pip install --no-cache-dir \
    "gymnasium[mujoco]" \
    "minari[hdf5]" \
    "imitation" \
    "stable-baselines3>=2.2.0" \
    imageio \
    matplotlib \
    numpy

# Set MuJoCo to use OSMesa for headless rendering
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Copy the script
COPY rl25_task3.py .

# We can also copy the notebook if the user wants it inside, but we run the script
COPY rl25_task3_firstname_secondname.ipynb .

CMD ["python", "rl25_task3.py"]
