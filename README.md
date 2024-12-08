
# Drone Simulation with Neural Network Integration

This project simulates a drone navigating towards targets within a 2D environment. The simulation is designed to integrate human control or neural network models trained by genetic algorithm for autonomous navigation.

## Features

- **Drone Dynamics**: Realistic physics for drone movement including acceleration, angular rotation, and thruster control.
- **Target Tracking**: The drone aims to reach randomly generated or pre-defined targets.
- **Neural Network Integration**: Supports custom-trained (with genetic algorithm) neural networks for autonomous navigation.
- **Visualization**: Real-time simulation using `pygame`.

---

## Getting Started

### Prerequisites

- Python 3.9 or later
- Virtual environment (optional, but recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/skublin/drone-ea.git
   cd drone-ea
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place any pre-trained models in the `models` directory.

---

## Running the Simulation

To run the simulation with a pre-trained neural network model:

```bash
python main.py --model models/model-1.h5
```

For manual control (keyboard):
```bash
python main.py
```

---

## Key Controls (Manual Mode)

| Key    | Action       |
|--------|--------------|
| `W`    | Increase thrust (both engines) |
| `S`    | Decrease thrust (both engines) |
| `A`    | Decrease left engine thrust    |
| `D`    | Decrease right engine thrust   |

---

## Files and Structure

- `main.py`: Entry point for running the simulation.
- `train.py`: Utilities for training or evaluating neural network models.
- `simulation.py`: Manages the game state and logic.
- `drone.py`: Handles drone physics and rendering.
- `target.py`: Defines target behavior and drawing.
- `settings.py`: Central configuration file for the simulation.

---

## Neural Network Integration

You can integrate custom neural networks trained to navigate the drone. The model should:
- Accept a 6-dimensional input vector:
  - `[dx, dy, vx, vy, angular_velocity, angle]`
- Output 2 values:
  - `[force_amplitude, force_difference]`

To use a custom model, save it in `.h5` or `.pkl` format and specify its path in `main.py`.

---

## Future Enhancements

- Add pre-trained models.
- Include reinforcement learning examples.
- Add more visual effects and drone states.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
