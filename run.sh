#!/bin/bash

# Script de ejecución para el proyecto de simulación SIR con Docker

echo "🚀 Ejecutando Simulación SIR con Docker Compose"
echo "==============================================="
echo ""

# Verificar que Docker esté instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker no está instalado"
    echo "Por favor instala Docker desde: https://www.docker.com/get-started"
    exit 1
fi

# Verificar que Docker Compose esté disponible
if ! docker compose version &> /dev/null; then
    echo "❌ Error: Docker Compose no está disponible"
    exit 1
fi

# Crear directorio de salida si no existe
mkdir -p output

# Ejecutar la simulación
echo "📊 Ejecutando simulación y visualización..."
docker compose up run_all

# Verificar el resultado
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Simulación completada exitosamente"
    echo ""
    echo "📁 Resultados guardados en: ./output/"
    echo "   - test_simulation_plot.png (Dinámica SIR)"
    echo "   - simple_sbm_comparison.png (Comparación de redes)"
    echo ""
    
    # Limpiar contenedores
    echo "🧹 Limpiando contenedores..."
    docker compose down
    
    echo ""
    echo "✨ ¡Listo! Revisa los archivos en la carpeta output/"
else
    echo ""
    echo "❌ Error durante la ejecución"
    exit 1
fi
