# Explanation of AI Surrogate Architectures (Explicación de las Arquitecturas de IA)

This document explains the models found in `ArchNN.txt` simply, with diagrams to help visualize them.
Este documento explica los modelos de `ArchNN.txt` de forma sencilla, con diagramas para visualizarlos mejor.

---

## 🇬🇧 English Explanation

These AI models act as "surrogates" (fast replacements) for a slow epidemic simulator. They take 4 initial parameters and predict the epidemic curve.

### 1. Basic Autoencoder (Basic AE)
- **What it does:** It compresses the starting parameters into a small summary (latent space), and then uncompresses them to draw the entire epidemic curve all at once.
- **Pros & Cons:** Very fast, but it doesn't "understand" the step-by-step passing of time.

```mermaid
graph TD
    In["Inputs (4 Parameters)"] --> Enc["Encoder (Compresses)"]
    Enc --> Latent["Latent Space (Summary)"]
    Latent --> Dec["Decoder (Uncompresses)"]
    Dec --> Out["Entire Curve I(t) drawn at once"]
    
    style In fill:#dbeafe,stroke:#1e3a8a
    style Enc fill:#dbeafe,stroke:#1e3a8a
    style Latent fill:#93c5fd,stroke:#1e3a8a
    style Dec fill:#dbeafe,stroke:#1e3a8a
    style Out fill:#dbeafe,stroke:#1e3a8a
```

### 2. Deep Autoencoder + Smoothing (Deep AE + Smooth)
- **What it does:** Similar to the basic one, but deeper/more complex. At the end, it uses a "smoothing filter".
- **Pros & Cons:** The smoothing filter prevents the predicted curve from having weird, sudden jumps.

```mermaid
graph TD
    In["Inputs (4 Parameters)"] --> Enc["Deep Encoder"]
    Enc --> Latent["Deep Latent Space"]
    Latent --> Dec["Deep Decoder"]
    Dec --> Smooth["Smoothing Filter"]
    Smooth --> Out["Stable Epidemic Curve"]

    style In fill:#ccfbf1,stroke:#0f766e
    style Enc fill:#ccfbf1,stroke:#0f766e
    style Latent fill:#5eead4,stroke:#0f766e
    style Dec fill:#ccfbf1,stroke:#0f766e
    style Smooth fill:#ffedd5,stroke:#c2410c
    style Out fill:#ccfbf1,stroke:#0f766e
```

### 3. Autoregressive LSTM (LSTM)
- **What it does:** It uses parameters to set up an initial "memory state". Then, it generates the curve day by day, using yesterday's data to predict today.
- **Pros & Cons:** It is the closest to how the real simulator works because it respects the step-by-step sequence of time.

```mermaid
graph TD
    In["Inputs (4 Parameters)"] --> Enc["Parameter Encoder"]
    Enc --> State["Initial State (Memory)"]
    State --> LSTM["LSTM Network"]
    LSTM --> Step["Step-by-step: Yesterday → Today"]
    
    style In fill:#f3e8ff,stroke:#6b21a8
    style Enc fill:#f3e8ff,stroke:#6b21a8
    style State fill:#d8b4fe,stroke:#6b21a8
    style LSTM fill:#f3e8ff,stroke:#6b21a8
    style Step fill:#f3e8ff,stroke:#6b21a8
```

---

## 🇪🇸 Explicación en Español

Estos modelos de IA actúan como "sustitutos" (reemplazos rápidos) de un simulador lento. Toman 4 parámetros e intentan predecir la curva de la epidemia.

### 1. Autoencoder Básico (Basic AE)
- **Qué hace:** Comprime los parámetros iniciales en un pequeño resumen (espacio latente) y luego los descomprime para dibujar toda la curva de una sola vez.
- **Pros y Contras:** Es muy rápido, pero no "entiende" el paso del tiempo día a día.

```mermaid
graph TD
    In["Entradas (4 Parámetros)"] --> Enc["Encoder (Comprime)"]
    Enc --> Latent["Espacio Latente (Resumen)"]
    Latent --> Dec["Decoder (Descomprime)"]
    Dec --> Out["Curva entera I(t) dibujada de golpe"]
    
    style In fill:#dbeafe,stroke:#1e3a8a
    style Enc fill:#dbeafe,stroke:#1e3a8a
    style Latent fill:#93c5fd,stroke:#1e3a8a
    style Dec fill:#dbeafe,stroke:#1e3a8a
    style Out fill:#dbeafe,stroke:#1e3a8a
```

### 2. Autoencoder Profundo + Suavizado (Deep AE + Smooth)
- **Qué hace:** Similar al básico, pero más grande y complejo. Al final, utiliza un "filtro de suavizado".
- **Pros y Contras:** El filtro evita que la curva predicha tenga saltos extraños y repentinos.

```mermaid
graph TD
    In["Entradas (4 Parámetros)"] --> Enc["Encoder Profundo"]
    Enc --> Latent["Espacio Latente Profundo"]
    Latent --> Dec["Decoder Profundo"]
    Dec --> Smooth["Filtro de Suavizado"]
    Smooth --> Out["Curva Epidémica Estable"]

    style In fill:#ccfbf1,stroke:#0f766e
    style Enc fill:#ccfbf1,stroke:#0f766e
    style Latent fill:#5eead4,stroke:#0f766e
    style Dec fill:#ccfbf1,stroke:#0f766e
    style Smooth fill:#ffedd5,stroke:#c2410c
    style Out fill:#ccfbf1,stroke:#0f766e
```

### 3. LSTM Autorregresivo (LSTM)
- **Qué hace:** Utiliza los parámetros iniciales para configurar un "estado de memoria" inicial. Luego, genera la curva día a día, utilizando los datos de ayer para predecir los de hoy.
- **Pros y Contras:** Es el que más se parece a cómo funciona el simulador real porque respeta la secuencia del tiempo paso a paso.

```mermaid
graph TD
    In["Entradas (4 Parámetros)"] --> Enc["Encoder de Parámetros"]
    Enc --> State["Estado Inicial (Memoria)"]
    State --> LSTM["Red LSTM"]
    LSTM --> Step["Paso a paso: Ayer → Hoy"]
    
    style In fill:#f3e8ff,stroke:#6b21a8
    style Enc fill:#f3e8ff,stroke:#6b21a8
    style State fill:#d8b4fe,stroke:#6b21a8
    style LSTM fill:#f3e8ff,stroke:#6b21a8
    style Step fill:#f3e8ff,stroke:#6b21a8
```
