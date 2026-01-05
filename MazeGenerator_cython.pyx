# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False, infer_types=True, language_level=3


import cython

import numpy as np
cimport numpy as np

np.import_array()
from libc.time cimport time

ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.float32_t float32

# simple xorshift32 RNG
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline unsigned int xorshift32(unsigned int* state) nogil:
        cdef unsigned int x = state[0]
        x ^= x << 13
        x ^= x >> 17
        x ^= x << 5
        state[0] = x
        return x

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef uint8[:, :, :]  grid_to_img(const uint8[:, :, ::1] grid):
        """
        Translates the 0-1 grid array into an RGB image
        We color the starting cell in green and the end cell in red
        """
        cdef unsigned char[:, :, ::1] img
        cdef int i
        cdef int j
        cdef Py_ssize_t grid_size = grid.shape[1]

        img = np.empty((grid_size, grid_size,3), dtype=np.uint8)
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[0,i,j] == 1:
                    img[i,j,0] = 255
                    img[i,j,1] = 255
                    img[i,j,2] = 255
                elif grid[1,i,j] == 1:
                    img[i,j,0] = 0
                    img[i,j,1] = 0
                    img[i,j,2] = 0
                elif grid[2,i,j] == 1:
                    img[i,j,0] = 0
                    img[i,j,1] = 255
                    img[i,j,2] = 0
                else:
                    img[i,j,0] = 255
                    img[i,j,1] = 0
                    img[i,j,2] = 0

        return img

cdef class MazeGenerator_cython():
    """
    Convention for one-hot encoding:
    (1,0,0,0) : blank
    (0,1,0,0) : wall
    (0,0,1,0) : starting cell
    (0,0,0,1) : end cell

    """
    cdef Py_ssize_t grid_size
    cdef int nb_step_end
    cdef public unsigned int seed


    cdef np.ndarray _stack_x_arr
    cdef np.ndarray _stack_y_arr
    cdef np.ndarray _stack_d_arr
    cdef np.ndarray _possible_x_arr
    cdef np.ndarray _possible_y_arr


    cdef int32[::1] stack_x, stack_y, stack_d
    cdef int32[::1] possible_end_x, possible_end_y

    def __cinit__(self, int grid_size, int nb_step_end, seed=None):
        """
        nb_step_end :
        the number of steps the agents needs to take to reach the end cell, if no cell satisfies this we take the farthest away from the starting cell

        """
        self.grid_size = grid_size
        self.nb_step_end = nb_step_end

        if seed is not None:
            self.seed = <unsigned int> seed
        else:
            self.seed = <unsigned int> time(NULL)
        if self.seed == 0:
            self.seed = 1


        self._stack_x_arr = np.empty(grid_size * grid_size, dtype=np.int32)
        self._stack_y_arr = np.empty(grid_size * grid_size, dtype=np.int32)
        self._stack_d_arr = np.empty(grid_size * grid_size, dtype=np.int32)

        cdef int p_cap = (2*nb_step_end+1)
        self._possible_x_arr = np.empty(p_cap*p_cap, dtype=np.int32)
        self._possible_y_arr = np.empty(p_cap*p_cap, dtype=np.int32)


        self.stack_x = self._stack_x_arr
        self.stack_y = self._stack_y_arr
        self.stack_d = self._stack_d_arr
        self.possible_end_x = self._possible_x_arr
        self.possible_end_y = self._possible_y_arr

    def __init__(self, int grid_size, int nb_step_end, seed = None):
        assert grid_size > 5 # we do not allow grids that are too small just to be safe
        assert grid_size % 2 == 1 # we use the depth-first algorithm that works in step of 2
        assert nb_step_end > 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int  _get_unvisited_neigbhours(self, int x , int y, const uint8[:,:,::1] grid, int32[4] neighbour_x, int32[4] neighbour_y) nogil:
        cdef int count = 0
        cdef Py_ssize_t gs = self.grid_size

        if x > 1 and grid[1,x-2,y] == 1:
            neighbour_x[count] = x-2
            neighbour_y[count] = y
            count+=1
        if x < gs-2 and grid[1,x+2,y] == 1:
            neighbour_x[count] = x+2
            neighbour_y[count] = y
            count+=1
        if y > 1 and grid[1,x,y-2] == 1:
            neighbour_x[count] = x
            neighbour_y[count] = y-2
            count+=1
        if y < gs-2 and grid[1,x,y+2] == 1:
            neighbour_x[count] = x
            neighbour_y[count] = y+2
            count+=1

        return count


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void generate(self, uint8[:, :, ::1] grid, int32[::1] start_cell, int32[::1] end_cell):
        """
        Generates a maze using the depth-first search algorithm
        """
        
        cdef int i,j,k
        cdef int gs = self.grid_size
        cdef int32[::1] stack_x = self.stack_x # stack used to hold the coordinates of the cells to visit in the DFS
        cdef int32[::1] stack_y = self.stack_y 
        cdef int32[::1] stack_d = self.stack_d # used to track the distance from the starting cell
        cdef int32[::1] possible_end_x = self.possible_end_x # cells that are at the right distance from the starting cell
        cdef int32[::1] possible_end_y = self.possible_end_y

        # initialize all cells to wall
        for i in range(gs):
            for j in range(gs):
                grid[0,i,j] = 0
                grid[1,i,j] = 1
                grid[2,i,j] = 0
                grid[3,i,j] = 0


        cdef int32[4] unvisited_neigbhours_x, unvisited_neigbhours_y
        cdef int unvisited_neigbhours_count
        cdef int x,y, ix, iy, nx, ny, idx, d
        cdef int d_max = -1
        cdef int d_max_x = -1
        cdef int d_max_y = -1
        cdef int top = 0
        cdef int possible_count = 0
        cdef int m = (gs-1)//2
        cdef unsigned int* rng_state = &self.seed


        x = 1 + 2*(xorshift32(rng_state) % m)
        y = 1 + 2*(xorshift32(rng_state) % m)
        start_cell[0] = x
        start_cell[1] = y

        grid[2,start_cell[0], start_cell[1]] = 1
        grid[1,start_cell[0], start_cell[1]] = 0
        stack_x[top] = start_cell[0]
        stack_y[top] = start_cell[1]
        stack_d[top] = 0


        while top >= 0:
            x = stack_x[top]
            y = stack_y[top]
            d = stack_d[top]
            unvisited_neigbhours_count = self._get_unvisited_neigbhours(x, y, grid, unvisited_neigbhours_x, unvisited_neigbhours_y)
            if unvisited_neigbhours_count == 0:
                top -= 1
            else:
                idx = xorshift32(rng_state) % unvisited_neigbhours_count
                nx = unvisited_neigbhours_x[idx]
                ny = unvisited_neigbhours_y[idx]
                d += 1
                ix = (x+nx)//2 # we take steps of 2 so we need to consider the intermediate cell
                iy = (y+ny)//2
                if d > d_max:
                    d_max = d
                    d_max_x, d_max_y = nx,ny
                if d == self.nb_step_end:
                    possible_end_x[possible_count] = nx
                    possible_end_y[possible_count] = ny
                    possible_count += 1
                grid[0,ix,iy] = 1
                grid[0,nx,ny] = 1
                grid[1,ix,iy] = 0
                grid[1,nx,ny] = 0
                top+=1
                stack_x[top] = nx
                stack_y[top] = ny
                stack_d[top] = d


        # if there is no accessible point that is far enough we pick the farthest one
        if possible_count == 0:
            end_cell[0], end_cell[1] = d_max_x, d_max_y
        else:
            idx = xorshift32(rng_state) % possible_count
            end_cell[0], end_cell[1]  = possible_end_x[idx], possible_end_y[idx]

        grid[3,end_cell[0], end_cell[1]] = 1
        grid[0,end_cell[0], end_cell[1]] = 0

        return
