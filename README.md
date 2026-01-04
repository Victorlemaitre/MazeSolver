
### TLDR:
- Used PPO to train an agent to solve small and fully observable maze at test time
- Implemented my own vectorized environement in cython and made it generates a new maze at each reset in order to force the agent to generalize
- Following the idea of curriculum training I gradually increased the distance between the start and the exit during training


---

### Motivation

Following my first semester at [master IASD](https://www.masteriasd.eu/en/) I felt like my course in RL was too theoretical hence I took the goal to implement some policy gradient algorithm like REINFORCE, A2C and PPO during the christmas holidays. 

Initially I implemented a small grid world in a custom gym environment where you would design the grid yourself by placing the agent starting position, the walls and the rewards. However I found that the task was made either trivial or almost impossible depending on the grid design. I also disliked the idea that the agent was not really reasoning but was merely overfitting to a particular grid design. To solve that last problem I would have need to generate a lot of new grids which I could not hence I set out to solve small maze as they can be efficiently generated.

### Environment description :

Each maze possess 4 types of cells: 
- Blanks
- Walls
- The agent
- The exit

At each step the agent is given the full maze as observation.

The rewards are : 
- +1 on reaching the exit
- -0.001 on bumping into a wall 
- -0.001 at each step to encourage efficient path

To generate the labyrinths I employed the depth-first search algorithm 

<p align="center">
  <img src="demo.gif" width="300" alt="Demo">
</p>
