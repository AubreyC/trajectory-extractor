# Carla Trajectories Generator

Generate trajectories ground truth with CARLA simulator:
- Ground truth csv with agent information - Position, Velocity, Orientation, etc
- Camera images from different location - Simulate surveillance caemra

## Installation the simulator

Build CARLA from source: https://carla.readthedocs.io/en/stable/how_to_build_on_linux/

## Generate trajectories

### Add camera to the simulation:

Start the Carla Simulator (in Unreal):

    cd Unreal/CarlaUE4
    ~/Documents/code/carla/UnrealEngine_4.18/Engine/Binaries/Linux/UE4Editor "$PWD/CarlaUE4.uproject"

Add camera:

- Add `SceneCaptureToDisk` object in the Carla World
- Change saving folder, images sizes, capture per seconds

Set the Simulation step:

- Edit -> Editor Preferencs -> Level Editor -> Play -> Play in Standalone Game -> Additional Launch Paramaters: `-benchmark -fps=15`

### Start the game

In Unreal:

- Play Standalone Game

Start the python script:

    python run_traj_gen.py