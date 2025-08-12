# DOOM-bots
Various reinforcement learning and evolutionary computation projects in ViZDOOM. 

For all models to properly run, you must place DOOM.WAD, DOOM2.WAD and freedoom2.wad in the 'Games' folder.
DOOM.wad and DOOM2.wad can be purchased on Steam or GOG, and freedoom2.wad can be found here: https://freedoom.github.io/download.html

Projects include:

1. Bot trained to efficiently complete the **Basic**, **Defend the Center** and **Deadly Corridor** scenerios using a CNN trained with visual information and with weights adjusted by the PPO algorithm. 
2. Genetic algorithm evolved to complete the **Defend the Center** scenerio while defeating the most amount of enemies. Fitness function based off total enemies defeated - missed shots. 
3. Bot trained to solve the **My Way Home** maze scenerio using the [Ant Colony Optimization](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms) algorithm.

Scenerios described can be found here: https://vizdoom.farama.org/environments/default/

# Setup

1. pip install -r requirements.txt 
2. Place wads in games folder
3. Run RL env and models in Reinforcement Learning/vizdoom_learning.py, GA env and model in Genetic Algorithm/vizdoom_defend_the_center_GA, ACO env and model in ACO Pathfinding/vizdoom_ACO_grid.ipynb