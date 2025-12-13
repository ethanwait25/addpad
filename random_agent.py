from agent import Agent
from random import randint

class RandomAgent(Agent):
    def act(self, obs):
        return randint(0, 13)