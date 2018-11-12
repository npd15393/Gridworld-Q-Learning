import numpy as np
import random
import math
import pygame, sys, time, random
from pygame.locals import *
from tile import Tile
from agent import Agent

class Env:
	tile_width = 100
	tile_height = 100
	def __init__(self, size):
		self.current_state=(0,0)
		# self.current_state=(0,0,0,4)
		self.goal=(4,4)
		self.actions=[(0,1),(0,-1),(1,0),(-1,0),(0,0)]
		self.n_agents=1
		self.all_R={}#*self.n_agents
		self.single_S=[]
		self.size=size
		self.S=[]
		self.obstacles=[(2,1),(2,2)]
		self.add_ss()
		# self.create_joint_A()
		self.set_reward()
		self.render=False
		self.deterministic=True
		
		self.start_renderer()
		

	def start_renderer(self):
		"""
		Sets up pygame objects
		"""

		# Init
		pygame.init()
		Tile.size = self.size
 
		# Set window size and title, and frame delay
		surfaceSize = (100*self.size[0], 100*self.size[1])
		windowTitle = 'Grid_World'
		self.pauseTime = 1  # smaller is faster game

		# Create the window
		self.surface = pygame.display.set_mode(surfaceSize, 0, 0)
		self.bgColor = pygame.Color('black')
		pygame.display.set_caption(windowTitle)

		# Init tile objects
		self.createTiles()


	def createTiles(self):
		"""
		Sets up tile object arrangement in the grid
		:param exp: Experience tuple from Env
		:return: void
		"""
		self.board = []
		for yIndex in range(self.size[1]):
			row = []
			for xIndex in range(self.size[0]):
				imageIndex = yIndex * self.size[1] + xIndex
				y = (self.size[1] -1- yIndex) * Env.tile_width
				x = xIndex * Env.tile_height
				if (xIndex,yIndex) in self.obstacles:
					wall = True
				else:
					wall = False
				tile = Tile(x, y, wall, self.surface)
				row.append(tile)
			self.board.append(row)

	def renderEnv(self,a=None):
		"""
		Update pygame visualization
		:param a: agent object
		"""
		print(self.current_state)

		for event in pygame.event.get():
				if event.type == QUIT:
					pygame.quit()
					sys.exit()

		self.draw(a)


		# Refresh the display
		pygame.display.update()

		# Set the frame speed by pausing between frames
		time.sleep(self.pauseTime)


	def draw(self,agent=None):	
		"""
		Draw the tiles.
		:param agent: used to query optimal action from agent object
		"""	
		pos=self.current_state
		goal = self.goal
		self.surface.fill(self.bgColor)

		for y in range(self.size[1]):
			for x in range(self.size[0]):
				if not agent==None:idx=self.actions.index(agent.get_optimal_a((x,y),True))
				row=self.board[y]
				row[x].draw(pos, goal,idx)


	def setTesting(self):
		"""
		Set testing mode on after training
		"""
		self.render=True
		self.deterministic=True

	def add_ss(self):
		"""
		Populates a list of valid states
		"""
		for x in range(self.size[0]):
			for y in range(self.size[1]):
				self.S.append((x, y))

	def deterministic_transition(self,state, action):
		"""
		This function return the next state based on deterministic transition without the transition probability.
		:param state: The current state
		:param action: The current action
		:return: The deterministic next state
		"""
		next_state = tuple(np.array(state) + np.array(action))
		if not self.check_s(next_state):
			next_state=state
		
		return tuple(next_state)

	def reset(self):
		while True:
			self.current_state=(random.randint(0, self.size[0] - 1),random.randint(0, self.size[1] - 1))
			if self.check_s(self.current_state):
				# self.renderEnv()
				return self.current_state

	def step(self,agent):

		act=agent.get_a_exp(self.current_state,self.deterministic)
		act_idx = self.actions.index(act)

		r=self.get_all_R(self.current_state,act)

		prev_st=self.current_state
		
		self.current_state=self.deterministic_transition(self.current_state,act)

		if self.render:
			self.renderEnv(agent)

		isDone=self.checkGoal(self.current_state)

		return [prev_st,act,self.current_state,r,isDone]

	def checkCollision(self, n_s):
		"""
		Check if the state n_s is colliding with obstacle
		:return: boolean
		"""
		return (n_s in self.obstacles)

	def checkBounds(self, n_s):
		"""
		Check if the state n_s is in the MDP
		:return: boolean
		"""
		flag = n_s[0] < self.size[0] and n_s[0] >= 0 and n_s[1] < self.size[1] and n_s[1] >= 0
		return flag

	def checkGoal(self,s):
		"""
		Check if s is goal
		:return: boolean
		"""
		flag=(s==self.goal)
		return flag

	def check_a(self, s, a):
		"""
		Check if action from a given state s is valid.
		:return: boolean
		"""
		ns = self.deterministic_transition(s, a)
		return check_s(ns)

	def check_s(self, s):
		"""
		Check if state is valid.
		:return: boolean
		"""
		return (not self.checkCollision(s)) and self.checkBounds(s)

	def set_reward(self):
		"""
		This function set positive and negative reward based on the requirements for the MDP.
		:return: NULL
		"""
		for s in self.S:
			for a in self.actions:
				n_state = self.deterministic_transition(s, a)
				for i in range(self.n_agents):
					if self.checkGoal(n_state):
						self.all_R[s, a] = 1
					elif self.checkCollision(n_state):
						self.all_R[s, a] = -1
					else:
						self.all_R[s, a] = -0.4
		return
	
	def get_all_R(self,s,a):
		"""
		Return reward for key (s,a)
		:return: Reward value
		"""
		return self.all_R[s,a]
