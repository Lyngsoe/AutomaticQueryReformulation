import numpy as np

class MovingAverageBasic:
    def __init__(self,start_value,window_size=1000):
        self.name = "MovingAverageBasic"
        self.base_reward = start_value
        self.window_size = window_size
        self.gamma = 1/self.window_size
    def __call__(self, *args, **kwargs):
        return self.base_reward
    def update(self, r):
        self.base_reward = self.gamma * r + (1 - self.gamma) * self.base_reward

class MovingAverage:
    def __init__(self,window_size=1000):
        self.name = "MovingAverage"
        self.base_reward = np.zeros(window_size)
        self.window_size = window_size
        self.iter = 0
        self.startup = 0
    def __call__(self,r):
        if self.startup == 0:
            return 0
        elif self.startup < self.window_size:
            return np.mean(self.base_reward[:self.startup])
        else:
            return np.mean(self.base_reward)
    def update(self, r):
        self.iter +=1
        self.startup+=1
        self.iter %= self.window_size
        self.base_reward[int(self.iter)] = r

class MovingStdAverage:
    def __init__(self,window_size=1000):
        self.name = "MovingStdAverage"
        self.base_reward = np.zeros(window_size)
        self.window_size = window_size
        self.iter = 0
        self.startup = 0
    def __call__(self, reward_lst):

        if self.startup == 0:
            base_reward = 0
        elif self.startup < self.window_size:
            base_reward = np.mean(self.base_reward[:self.startup])
        else:
            base_reward = np.mean(self.base_reward)

        normalized_reward = []
        for r in reward_lst:
            if self.startup == 0:
                normalized_reward.append(r)
            elif self.startup < self.window_size:
                if base_reward == 0:
                    normalized_reward.append(r)
                else:
                    normalized_reward.append((r - base_reward) / np.std(self.base_reward[:self.startup]))
            else:
                normalized_reward.append((r - base_reward) / np.std(self.base_reward))
        return base_reward, normalized_reward

    def update(self, r):
        self.iter +=1
        self.startup+=1
        self.iter %= self.window_size
        self.base_reward[int(self.iter)] = r