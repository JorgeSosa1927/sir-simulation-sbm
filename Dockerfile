# Usa una imagen base de Python 3.11
FROM python:3.11-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Instala las dependencias del sistema necesarias para matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia el archivo de requisitos (lo crearemos a continuación)
COPY requirements.txt .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos los archivos del proyecto al contenedor
COPY *.py .

# Crea un directorio para los resultados
RUN mkdir -p /app/output

# Comando por defecto: ejecutar la simulación
CMD ["python", "test_simulation.py"]
