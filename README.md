
### TLDR:
Personal learning project where I trained an agent to solve small, fully observable procedurally generated mazes at test time. To do so I implemented PPO from scratch in pytorch. I also implemented my own vectorized environment in cython and made it generate a new maze at each reset in order to force the agent to generalize. Finally I used ideas from curriculum training to fight the natural reward sparsity of the task. 



---
*Please note that the agent has never seen any of those maze during training*
<p align="center">
  <img src="Assets/Maze13_solve.gif" width="600" alt="Maze13_solve">
</p>



### Motivation

Following my first semester in the [Master IASD](https://www.masteriasd.eu/en/) program, I felt like my course in RL was too theoretical hence I took the goal to implement policy gradient algorithms such as REINFORCE, A2C and PPO during the Christmas holidays. 

Initially I implemented a small grid world in a custom gym environment where you would design the grid yourself by placing the agent starting position, the walls and the rewards. However I found that the task was made either trivial or almost impossible depending on the grid design. I also disliked the idea that the agent was not reasoning but was merely overfitting to a particular grid design. To solve that last problem I would have need to generate a lot of new grids which I could not therefore I set out to solve small maze as they can be efficiently generated.

### Environment description :

Each maze possesses 4 types of cells: 
- Blanks
- Walls
- The agent
- The exit

At each step the agent is given the full maze as observation where each type of cell is one-hot encoded.

The rewards are : 
- +1 on reaching the exit
- -0.001 on bumping into a wall 
- -0.001 at each step to encourage efficient path

Positive rewards are very sparse and the other rewards are not directly useful to find the right path. This made training much more tedious than what it could have been but it was deliberate as I wanted the agent to learn its own set of heuristics without explicit guidance.

To generate the labyrinths I employed the depth-first search algorithm which you can see in action below :

<p align="center">
  <img src="Assets/DFS_maze_gif.gif" width="400" alt="DFS maze">
</p>

The starting position is placed randomly and the exit is located at a fixed distance from the start. The distance between exit and start can be controlled to manipulate the difficulty of the maze which is what I will be doing during training.

To force generalization a new maze is generated at every reset.

To generate the environment I started with gymnasium and used its vectorized utilities. However I realized that speed was going to be an issue during training. As a result I stepped out of gymnasium entirely and implemented everything in cython including the handling of vectorized environments. This brought a few change to the training loop. For instance, now for performance purposes the observations, rewards are updated in place. These optimizations made the simulation cost negligible (On my average laptop it can run at well above 100K steps/second). To give you an idea of the effectiveness of cython, the pure python maze generation algorithm can generate about 3K/second $13\times 13$ mazes while the cython version can generate 600K/second of them.
Now most of the runtime during training is dedicated to updating the agent's weigths as well as CPU-GPU data transfers.

### Training algorithm :

I used PPO which I implemented in pytorch inspiring myself from  Clean RL's [implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py). Handling the mask logic in GAE as well as the reset of vectorized environments was quite tricky and not something I'd like to redo during holidays but alas it is done now and the agent is learning. 

During training I kept track of the average length needed for the agent to reach the exit. If it was close to perfect I incremented the distance between exit and start. This helped alleviate the sparsity of the rewards.

### Agent architecture :

Backbone (CNN encoder):
- Conv2d(nb_channel → 32, kernel=3, padding=0) 
- ReLU
- Conv2d(32 → 64, kernel=3, padding=1)
- ReLU
- Conv2d(64 → 128, kernel=3, padding=1)
- ReLU
- Flatten
- Linear((maze_size-2)² × 128 → final_hidden_size)
- ReLU
- Linear(final_hidden_size → final_hidden_size)
- ReLU

Actor head:
- Linear(final_hidden_size → 4)

Critic head:
- Linear(final_hidden_size → 1)

### Results and limitations :

As you can see in the first gif of this README I was able to train an agent to solve maze of size $13\times 13$ at test time. However I found that training was heavily impacted by the dimensions of the maze. At higher maze size the computational cost of processing large image was becoming too much. Indeed as labyrinths's pixels have a lot of meaning downsizing the image size using higher strides in the convolutions results in worse performance. Moreover bigger maze also means that you train with longer trajectories that requires PPO to process larger inputs. 

Looking back I realize that maybe policy gradient algorithms or just CNN were not the best fit for this tasks as their inductive bias do not align well with the highly structured nature of labyrinths. 

### Notes on this repo :
This project required a GPU and I do not have one. As a result I used google collab and a single jupyter notebook for most of my experiments. Therefore the repo you see here is just that single notebook splitted into different file for clarity.  





