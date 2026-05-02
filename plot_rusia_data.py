import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = 'Data_Rusia_2022.csv'
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo {csv_path}")
        return

    # Leer los datos
    df = pd.read_csv(csv_path)

    # Limpiar la columna 'Casos nuevos' (quitar comas y convertir a número)
    df['Casos nuevos'] = df['Casos nuevos'].astype(str).str.replace(',', '').astype(float)

    # Convertir 'Fecha' a formato de fecha para graficar correctamente
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    # Crear la gráfica
    plt.figure(figsize=(12, 6))
    plt.plot(df['Fecha'], df['Casos nuevos'], color='firebrick', linewidth=2, marker='o', markersize=4)

    # Añadir títulos y etiquetas
    plt.title('Nuevos Casos Registrados en Rusia (Comienzos de 2022)', fontsize=15, pad=15)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Cantidad de Casos Nuevos', fontsize=12)

    # Formato de la gráfica
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Guardar la gráfica
    output_path = 'plot_rusia_2022.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfica generada exitosamente y guardada como: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    main()
