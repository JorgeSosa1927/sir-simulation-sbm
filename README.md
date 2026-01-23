# SIR Simulation with Docker Compose

This project contains an SIR epidemiological simulation based on a Stochastic Block Model (SBM) with hub projection.

## 📋 Prerequisites

- Docker installed ([Download Docker](https://www.docker.com/get-started))
- Docker Compose installed (included with Docker Desktop)

## 🚀 Quick Start

### Option 1: Run Full Simulation and Visualization

```bash
docker-compose up run_all
```

This command will execute:
1. The SIR simulation (`test_simulation.py`)
2. The graph visualization (`visualize_simple_sbm.py`)
3. Save the results in the `output/` folder

### Option 2: Run Simulation Only

```bash
docker-compose up simulation
```

### Option 3: Run Visualization Only

```bash
docker-compose up visualization
```

## 📊 Results

Results will be saved in the `output/` folder:

- **test_simulation_plot.png**: SIR dynamics plot (Susceptible, Infected, Recovered)
- **simple_sbm_comparison.png**: Visual comparison between the original network with hubs and the projected network

## 🛠️ Project Structure

```
toy_sir_simulation_patch/
├── Dockerfile                      # Docker container configuration
├── docker-compose.yml             # Service orchestration
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── model_output.py                # Model output class
├── simple_sbm_generator.py        # SBM generator and SIR simulation
├── test_simulation.py             # Test and simulation script
├── visualize_simple_sbm.py        # Visualization script
└── output/                        # Results folder (created automatically)
```

## 🔧 Configuration

Simulation parameters can be modified in `test_simulation.py`:

- **Network parameters**: Block sizes, mixing matrix
- **Epidemiological parameters**: `beta_network`, `beta_household`, `delta`
- **Initialization parameters**: Initial infected fraction, mobile bias

## 🧹 Cleanup

To remove containers and clean up:

```bash
docker-compose down
```

To also remove built images:

```bash
docker-compose down --rmi all
```

To clean the results folder:

```bash
rm -rf output/
```

## 📖 Model Description

This model implements:

1. **SBM Network Generation**: Social blocks, non-social blocks, and hubs
2. **Hub Projection**: Hubs are projected as connections between "manzanas" (blocks/neighborhoods)
3. **SIR Simulation**:
   - Stratified population (mobile and static)
   - Transmission within households and between nodes
   - Noisy weights on edges
4. **Visualization**: Side-by-side comparison of the original vs. projected network

## 📝 Notes

- The matplotlib backend is set to `Agg` to allow graph generation without a GUI.
- Results are automatically saved to the `output/` folder.
- Random seed is set for reproducibility.

## 🐛 Troubleshooting

If you encounter errors:

1. **Permission Error**: Ensure Docker has permissions to create files.
2. **Port in use**: Services do not use ports, so this shouldn't be an issue.
3. **Insufficient Memory**: Increase memory allocated to Docker in settings.

## 👨‍💻 Development

To make changes to the code:

1. Modify `.py` files locally
2. Rebuild the image: `docker-compose build`
3. Run again: `docker-compose up run_all`
