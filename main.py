import time
from env import Env
from agent import Agent

# Init
env=Env((5,6))
a1=Agent(env)
env.reset()
step_cnt=0

# Train
for ep in range(1000):
	exp=env.step(a1)
	step_cnt=step_cnt+1
	a1.update(exp)

	if step_cnt==10 or exp[-1]==True:
		env.reset()
		step_cnt=0

	print("Episode: {}".format(ep))


# Test run trained policy
env.reset()
env.setTesting()
isDone=False

while not isDone:
	_,_,_,_,isDone=env.step(a1)
	time.sleep(0.3)
