import numpy as np
import collections


class Grid:

    COORDINATE_SYSTEMS = ["1D", "2D", "3D"]

    def __init__(self, coordinate_system):
        # In this problem we map the specific D-dimensional problem to
        # the one dimensional one. To take care of the dimensionality
        # we include the volume factor in the vertex coefficients

        # contains grid elements
        self.arrays = {}
        # contains dk
        self.arrays_diff = {}
        # contains the name of the coordinate system
        self.coordinate_system = coordinate_system

    def init1d(self, name, grid_min, grid_max, grid_step):
        # creates 1 dimensional grid graph with equidistant spacing
        # and adds this array to the hash table self.arrays[name]

        grid_1d = np.arange(grid_min, grid_max + grid_step, grid_step)
        self.arrays_diff[name] = grid_step
        self.arrays[name] = grid_1d

    def return_array1d(self, name):
        # with a given key returns a corrsponding 1 dimentional array
        return self.arrays[name]

    def function_prod(self, list_of_unit_vectors, list_of_functions):
        # calculates a function on a grid when function is a product of function
        # acting independently on each coordinate
        # example:  list_of_unit_vectors = [x1, x2]
        # assume that f = f1 (x1) * f2 (x2)
        # then we can use: list_of_functions = [f1, f2, ...]
        # NOTE_ys: do we need to generalize it?

        # check that the order of keys in the list_of_unit_vectors is ordered
        # the same way as in self.arrays.keys
        # otherwise throw an error
        n_counter = collections.Counter(list_of_unit_vectors)
        s_counter = collections.Counter(self.arrays.keys())
        if n_counter != s_counter:
            print('INVALID LIST OF NAMES')
            return

        outer_mat = list_of_functions[0](self.arrays[list_of_unit_vectors[0]])

        if len(list_of_unit_vectors) == 1:
            return outer_mat

        for ind, name in enumerate(list_of_unit_vectors[1:]):
            temp = list_of_functions[ind + 1](self.arrays[name])
            outer_mat = np.outer(outer_mat, temp)

        return outer_mat.reshape(outer_mat.size)

    def jacobian(self):
        # creates a jacobian prefactor for a given coordinate_system

        list_of_unit_vectors = list(self.arrays.keys())

        coordinate_system = self.coordinate_system

        if coordinate_system == "1D":
            # To be fixed!
            list_of_functions = [lambda k: (2 * np.pi) ** (-2) * k ** 2]
        if coordinate_system == "2D":
            # To be fixed!
            list_of_functions = [lambda k: (2 * np.pi) ** (-2) * k ** 2]
        if coordinate_system == "3D":
            list_of_functions = [lambda k: (np.sqrt(2) * np.pi) ** (-1) * k ]

        output = self.function_prod(list_of_unit_vectors, list_of_functions)

        return output

    def dv(self):
        # creates a volume element for integration

        list_of_unit_vectors = list(self.arrays.keys())

        # create grid_diff from dk
        # this part creates the simplest integation cheme
        grid_diff = self.arrays_diff[list_of_unit_vectors[0]] * np.ones(len(self.arrays[list_of_unit_vectors[0]]))
        # for comparison purposes it is turned off  
        #grid_diff[0] = 0.5 * grid_diff[0]
        #grid_diff[-1] = 0.5 * grid_diff[-1]

        if(len(list_of_unit_vectors) == 1):
            return grid_diff

        for ind, name in enumerate(list_of_unit_vectors[1:]):
            temp_grid_diff = self.arrays_diff[name] * np.ones(len(self.arrays[name]))
            temp_grid_diff[0] = 0.5 * temp_grid_diff[0]
            temp_grid_diff[-1] = 0.5 * temp_grid_diff[-1]
            grid_diff = np.outer(grid_diff, temp_grid_diff)

        return grid_diff.reshape(grid_diff.size)

    def size(self):
        # this method returns the number of grid points
        list_of_unit_vectors = list(self.arrays.keys())
        grid_size = 1
        for unit in list_of_unit_vectors:
            grid_size = grid_size * len(self.arrays[unit])
        return grid_size
