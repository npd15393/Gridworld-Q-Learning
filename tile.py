import pygame, sys, time, random
from pygame.locals import *

class Tile:
	# An object in this class represents a single Tile that
	# has an image

	# initialize the class attributes that are common to all
	# tiles.

	borderColor = pygame.Color('black')
	borderWidth = 4  # the pixel width of the tile border
	image1 = pygame.image.load('turtlebot.jpg')
	image3 = pygame.image.load('charger.jpg')

	# policy=[pygame.image.load('0.jpg'),pygame.image.load('arrow-d.jpg'),pygame.image.load('arrow-l.jpg'),pygame.image.load('arrow-r.jpg'))
	size = (5,5)

	def __init__(self, x, y, obs, surface, tile_size = (100,100)):
		# Initialize a tile to contain an image
		# - x is the int x coord of the upper left corner
		# - y is the int y coord of the upper left corner
		# - image is the pygame.Surface to display as the
		# exposed image
		# - surface is the window's pygame.Surface object

		self.obs = obs
		self.origin = (x, y)
		self.tile_coord = (x//tile_size[0], y//tile_size[1])
		self.surface = surface
		self.tile_size = tile_size

	def grid2board(self,pos):
		board_coord=()
		for i in range(len(pos)//2):
			board_coord=board_coord+(pos[2*i], Tile.size[1]-1-pos[2*i+1])
		return board_coord

	def draw(self, pos, goal,idx=None):
		# Draw the tile.
		pos=self.grid2board(pos)
		goal=self.grid2board(goal)

		rectangle = pygame.Rect(self.origin, self.tile_size)
		if self.obs:
			pygame.draw.rect(self.surface, pygame.Color('black'), rectangle, 0)
		elif goal == self.tile_coord:
			pygame.draw.rect(self.surface, pygame.Color('green'), rectangle, 0)
		elif goal == self.tile_coord:
			pygame.draw.rect(self.surface, pygame.Color('blue'), rectangle, 0)
		else:
			pygame.draw.rect(self.surface, pygame.Color('white'), rectangle, 0)
			if not idx==None:self.surface.blit(pygame.image.load(str(idx)+'.jpg'), self.origin)

		if goal == self.tile_coord:
			self.surface.blit(Tile.image3, self.origin)

		if pos == self.tile_coord:
			self.surface.blit(Tile.image1, self.origin)

		pygame.draw.rect(self.surface, Tile.borderColor, rectangle, Tile.borderWidth)
