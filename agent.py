import math

class Agent:

	def __init__(self,env):
		self.Q={}
		self.C={}
		self.pi={}

		self.gamma=0.95
		self.alpha=lambda d: 0.8
		self.actions=env.actions
		self.t=0

	def get_a_exp(self,state,det=None):
		"""
		This function will return an action given a stationary policy given the current state.
		:param state: The current state
		:param det: Whether action is deterministic
		:return: The action
		"""
		if det==None:
			det=False

		return self.get_optimal_a(state,det)

	def get_optimal_a(self, state, det=None):
		"""
		This function will return an action given a stationary policy given the current state.
		:param state: The current state
		:param policy: The current policy
		:return: The action
		"""
		if det==None: det=False
		s = tuple(state)
		probs = []
		if not det:
			for a in self.actions:
				# a = tuple(action)
				probs.append(self.get_pi(s,a,self.pi))

			np.seterr(all='raise')
			try:
				probs = probs / np.linalg.norm(probs, ord=1)
			except Exception:
				print('ERROR')
				print('State' + str(state))
				print(probs)

			index = np.random.choice(range(len(self.actions)), p=probs)
			a_star = self.actions[index]
		else:
			for a in self.actions:
				# a = tuple(action)
				probs.append(self.get_pi(s,a,self.pi))
			a_star = self.actions[probs.index(max(probs))]
		return a_star



	def get_pi(self,s,a,func=None):
		"""
		Generic getter function for all policy dictionaries
		:param state: The current state
		:param action: The current action
		:return: policy
		"""
		if func==None:
			func=self.pi
		if (s,a) in func.keys():
			return func[s,a]
		else:
			return 1/len(self.actions)

	def get_vals(self,s,a,func):
		"""
		Generic getter function for all dictionaries with keys (s,a)
		:param state: The current state
		:param action: The current action
		:return: value
		"""
		if (s,a) in func.keys():
			return func[s,a]
		else:
			return 0

	def get_val(self,s,func):
		"""
		Generic getter function for all dictionaries with keys (s)
		:param state: The current state
		:param action: The current action
		:return: value
		"""
		if s in func.keys():
			return func[s]
		else:
			return 0

		
	def update(self, exp):
		"""
		Update Q function and policy
		:param exp: Experience tuple from Env
		:return: void
		"""
		s=exp[0]
		a=exp[1]
		s_=exp[2]
		r=exp[3]
		done=exp[4]

		self.t=self.t+1

		# q update
		n_keys=[(s_,act) for act in self.actions]
		q_n=[self.get_vals(k[0],k[1],self.Q) for k in n_keys]
		
		q_target=(r+self.gamma*max(q_n)) if not done else r
		# print('prev q '+str((s,a))+' : '+str(self.get_vals(s,a,self.Q)))
		self.Q[s,a]=self.get_vals(s,a,self.Q)+self.alpha(self.t)*(q_target-self.get_vals(s,a,self.Q))
		# print('new q '+str((s,a))+' : '+str(self.get_vals(s,a,self.Q)))
		keys=[(s,act) for act in self.actions]
		tot_q=sum([math.exp(self.get_vals(k[0],k[1],self.Q)) for k in keys])
		for k in keys:
			self.pi[k]=math.exp(self.get_vals(k[0],k[1],self.Q))/tot_q
