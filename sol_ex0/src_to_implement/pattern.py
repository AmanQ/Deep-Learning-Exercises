import numpy as np
import matplotlib.pyplot as plt



class Checker:
	def __init__(self, resolution, tile_size):
		self.resolution = resolution
		self.tile = tile_size
		self.output = None

	def draw(self):
		dim = self.resolution // (self.tile * 2)
		tile_shape = (self.tile, self.tile)
		tile_block = np.concatenate((
			np.concatenate( (np.zeros(shape=tile_shape, dtype=int),  np.ones(shape=tile_shape, dtype=int)), axis=0), #creating black and white 1 single row
			np.concatenate((np.ones(shape=tile_shape, dtype=int), np.zeros(shape=tile_shape, dtype=int)), axis=0) #creating whitw and black 1 single row
		), axis=1)

		board = np.tile(tile_block, (dim, dim)) #duplicating the rows to form checker

		self.output = board

		return board.copy()

	def show(self):
		plt.imshow(self.output, cmap="gray")
		plt.show()

class Circle:
    def __init__(self,resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        x, y = np.meshgrid(x, y)
        self.output = self.circle_gen(self.position, x, y)
        return self.output.copy()

    def circle_gen(self, position, x, y):
        circle_area = np.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2)
        circle_area[circle_area <= self.radius] = 1 #white region
        circle_area[circle_area > self.radius] = 0 #black region
        return circle_area


    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()



class Spectrum:
    def __init__(self,resolution):
        self.resolution = resolution
    
    def draw(self):
        spectrum_output=np.zeros([self.resolution,self.resolution, 3]) # n x n matrix with 3 values(R,G,B) in it
        spectrum_output[:,:,0]= np.linspace(0,1,self.resolution) # for blue value decreasing from left to right
        spectrum_output[:,:,1]=np.linspace(0,1,self.resolution).reshape(self.resolution, 1) #for green top to down value increasing, reshape beacuse values changing for rows for others column values are changing
        spectrum_output[:,:,2]= np.linspace(1,0,self.resolution) # for red value decreasing from right to left
        self.output = spectrum_output
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()



