# Evolutionary Drone Simulation

This project is a 2D simulation of a drone (UAV) navigating towards a target using evolutionary algorithms. The simulation is built using Python and leverages libraries such as Pygame and Pymunk to create a visual representation of the drone's movement and control in a two-dimensional space.

## Features
- **Drone Navigation**: The drone moves towards a target point in the 2D environment, adapting its path to reach the goal.
- **Interactive Simulation**: The user can click to set a new target for the drone in real-time, allowing for interactive demonstrations of the drone's behavior.
- **Evolutionary Algorithm (Upcoming)**: Future iterations of this project will include an evolutionary algorithm to improve the control logic for the drone.

## Project Structure
- `Simulation.py`: Contains the `Simulation` class, which manages the main loop of the simulation, including event handling, updates, and rendering.
- `Drone.py`: Contains the `Drone` class, which manages the drone's properties and movement logic.
- `Target.py`: Contains the `Target` class, which represents the target location that the drone attempts to reach.

## Requirements
The required Python libraries are listed in the `requirements.txt` file. To install them, run the following command:

```sh
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:
   ```sh
   git clone https://github.com/skublin/drone-ea.git
   ```
2. Navigate to the project directory:
   ```sh
   cd drone-ea
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the simulation:
   ```sh
   python Simulation.py
   ```

## Future Improvements
- **Evolutionary Control Algorithms**: Implement evolutionary algorithms to optimize the drone's pathfinding and target tracking capabilities.
- **Advanced Physics**: Utilize Pymunk for more realistic physics simulations, such as obstacle avoidance and more complex movement patterns.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to enhance the simulation or add new features.

## License
This project is licensed under the MIT License.


