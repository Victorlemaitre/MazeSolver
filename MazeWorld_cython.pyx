# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False, infer_types=True, language_level=3


import cython
from MazeGenerator_cython import MazeGenerator_cython
import numpy as np
cimport numpy as np

np.import_array()
from libc.time cimport time

ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.float32_t float32
cdef class MazeWorld_cython():
    cdef int num_envs, maze_size
    cdef public int max_step
    cdef public int current_step
    cdef int n_steps_per_update


    cdef np.ndarray _agent_loc_arr
    cdef np.ndarray _end_loc_arr
    cdef public np.ndarray _grid_arr
    cdef public np.ndarray _step_count_arr


    cdef public object maze_generator
    cdef object _infos

    cdef uint8[:,::1] terminated
    cdef uint8[:,::1] truncated
    cdef int32[:,::1] agent_loc
    cdef int32[:,::1] end_loc
    cdef uint8[:,:,:,::1] grid
    cdef float32[:,::1] reward
    cdef int32[::1] step_count




    def  __cinit__(self, int num_envs, int maze_size, int init_nb_step_end, int max_step, int n_steps_per_update, uint8[:,::1] terminated, uint8[:,::1] truncated, float32[:,::1] reward):
        self.num_envs = num_envs
        self.maze_size = maze_size
        self.max_step = max_step
        self.current_step = 0
        self.n_steps_per_update = n_steps_per_update
        self.maze_generator = MazeGenerator_cython(maze_size, init_nb_step_end)
        self.terminated = terminated # we update the terminated, truncated and reward arrays in place
        self.truncated = truncated
        self.reward = reward

        self._agent_loc_arr = np.empty((num_envs, 2), dtype=np.int32)
        self._end_loc_arr = np.empty((num_envs, 2), dtype=np.int32)
        self._grid_arr = np.empty((num_envs, 4, maze_size, maze_size), dtype=np.uint8) # Channel first for Conv2d later down the road
        self._step_count_arr = np.empty(num_envs, dtype=np.int32)

        self.agent_loc = self._agent_loc_arr
        self.end_loc = self._end_loc_arr
        self.grid = self._grid_arr
        self.step_count = self._step_count_arr

    def __init__(self, int num_envs, int maze_size, int init_nb_step_end, int max_step, int n_steps_per_update, uint8[:,::1] terminated, uint8[:,::1] truncated, float32[:, ::1] reward):
        super().__init__()
        assert num_envs > 0
        assert maze_size > 5
        assert init_nb_step_end > 0
        assert max_step > 1
        assert n_steps_per_update > 0
        assert terminated.shape[0] == n_steps_per_update and terminated.shape[1] == num_envs and terminated.ndim == 2 # we disabled the boundary checks so we need to be extra careful
        assert truncated.shape[0] == n_steps_per_update and truncated.shape[1] == num_envs and truncated.ndim == 2
        assert reward.shape[0] == n_steps_per_update and reward.shape[1] == num_envs and reward.ndim == 2

    cpdef void set_difficulty(self, int nb_end_steps):
        self.maze_generator = MazeGenerator_cython(self.maze_size, nb_end_steps)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void reset(self):
        cdef int e, step
        for e in range(self.num_envs):
            self.step_count[e] = 0
            self.maze_generator.generate(self.grid[e], self.agent_loc[e], self.end_loc[e]) # set a new maze for the agent to learn on
            for step in range(self.n_steps_per_update):
                self.reward[step, e] = <float32> 0
                self.terminated[step, e] = False
                self.truncated[step, e] = False


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void reset_partial(self, int e):
        self.step_count[e] = 0
        self.reward[self.current_step, e] = <float32> 0
        self.maze_generator.generate(self.grid[e], self.agent_loc[e], self.end_loc[e])


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void step(self, int32[::1] action):

        cdef int e
        cdef int nx,ny
        cdef int x,y
        cdef int current_step = self.current_step # idx used to update in place the terminated, truncated and reward arrays. To understand its role look at the training loop
        cdef uint8[:,::1] terminated = self.terminated
        cdef uint8[:,::1] truncated  = self.truncated
        cdef float32[:,::1] reward   = self.reward
        cdef uint8[:,:,:,::1] grid   = self.grid
        cdef int32[:,::1] agent_loc  = self.agent_loc
        cdef int32[:,::1] end_loc    = self.end_loc
        cdef int32[::1] step_count   = self.step_count
        cdef bint hit_a_wall
        cdef int prev_step

        for e in range(self.num_envs):
            terminated[current_step, e] = False
            truncated[current_step, e] = False
            hit_a_wall = False
            prev_step = (current_step - 1)%self.n_steps_per_update
            # we delay the reset of step_count by one step to record episode length
            if (truncated[prev_step, e] or terminated[prev_step, e]):
                self.reset_partial(e)
                continue

            step_count[e] += 1
            x = agent_loc[e,0]
            y = agent_loc[e,1]

            if action[e] == 0:
                nx = x+1
                ny = y
            elif action[e] == 1:
                nx = x-1
                ny = y
            elif action[e] == 2:
                nx = x
                ny = y + 1
            else:
                nx = x
                ny = y - 1

            # check if there is a wall
            if grid[e,1,nx,ny] == 1:
                reward[current_step, e] = <float32> -0.002
                hit_a_wall = True
            elif nx == self.end_loc[e,0] and ny == self.end_loc[e,1]: # check if the agent reached the end of the labyrinth
                reward[current_step, e] = <float32> 1
                terminated[current_step, e] = True
            else:
                reward[current_step, e] = <float32> -0.001 # small negative reward to encourage efficiency

            if step_count[e] >= self.max_step:
                truncated[current_step, e] =  True

            # update the position of the agent if the move is legal
            if not hit_a_wall:
                agent_loc[e,0] = nx
                agent_loc[e,1] = ny
                grid[e,2,x,y] = 0
                grid[e,0,x,y] = 1
                grid[e,2,nx,ny] = 1
                grid[e,0,nx,ny] = 0
        self.current_step = (current_step + 1) % self.n_steps_per_update
