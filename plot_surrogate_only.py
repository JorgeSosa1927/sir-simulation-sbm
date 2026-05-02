import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Importamos lo necesario del archivo principal
from AI_SBM import (
    OUTPUT_DIR,
    DATASET_FILE,
    MODEL_FILE,
    EpidemicSurrogateNet,
    plot_ultimate_validation
)

def main():
    print("=== Iniciando solo la generación de gráfica ===")
    
    dataset_file = DATASET_FILE
    if not os.path.exists(dataset_file):
        print(f"Error: No se encontró el dataset en {dataset_file}")
        return
        
    print("Cargando dataset para ajustar los normalizadores (scalers)...")
    data = np.load(dataset_file)
    X = data['X']
    Y = data['Y']
    
    # Ajustar scalers igual que en el entrenamiento
    X_scaler = StandardScaler()
    Y_scaler = MinMaxScaler()
    X_scaler.fit(X)
    Y_scaler.fit(Y)
    
    print("Cargando modelo pre-entrenado...")
    model = EpidemicSurrogateNet(input_dim=X.shape[1], output_dim=Y.shape[1])
    model_path = MODEL_FILE
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return
        
    # Cargar los pesos del modelo
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Generando gráfica con los nuevos parámetros base...")
    # Llamamos a la función que ya actualizamos con tus valores factor = 1.09
    plot_ultimate_validation(model, X_scaler, Y_scaler)
    
    print("=== Gráfica generada con éxito ===")

if __name__ == "__main__":
    main()
