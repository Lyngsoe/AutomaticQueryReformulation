import numpy as np

class MovingAverageBasic:
    def __init__(self,start_value,window_size=1000):
        self.name = "MovingAverageBasic"
        self.base_reward = start_value
        self.window_size = window_size
        self.gamma = 1/self.window_size
    def __call__(self, *args, **kwargs):
        return self.base_reward
    def update(self, r,q_ids):
        self.base_reward = self.gamma * r + (1 - self.gamma) * self.base_reward

class MovingAverage:
    def __init__(self,window_size=1000):
        self.name = "MovingAverage"
        self.base_reward = np.zeros(window_size)
        self.window_size = window_size
        self.iter = 0
        self.startup = 0
    def __call__(self,r,q_ids):
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
    def __call__(self, reward_lst,q_ids):

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

    def update(self, r,q_ids):
        self.iter +=1
        self.startup+=1
        self.iter %= self.window_size
        self.base_reward[int(self.iter)] = r


class MovingQuestionStdAverage:
    def __init__(self,window_size=10):
        self.name = "MovingStdAverage"
        self.base_reward = {}
        self.window_size = window_size
        self.delta = 0.0001

    def __call__(self, reward_lst,q_ids):

        normalized_reward = []
        base_reward = []
        for i in range(len(reward_lst)):
            base_r,norm_r = self._calc_base_reward(reward_lst[i],q_ids[i])
            normalized_reward.append(norm_r)
            base_reward.append(base_r)

        return base_reward,normalized_reward

    def _calc_base_reward(self,r,q_id):

        cur_rewards = self.base_reward.get(q_id,[])
        number_of_rewards = len(cur_rewards)

        if number_of_rewards  == 0:
            base_reward = r
        else:
            base_reward = np.mean(cur_rewards)

        r-= self.delta

        if number_of_rewards == 0:
            normalized_reward = r
        elif number_of_rewards == 1:
            normalized_reward = (r - base_reward)
        else:
            std = np.std(cur_rewards)
            if std == 0:
                std = 1
            normalized_reward = (r - base_reward) / std

        return base_reward, normalized_reward

    def update(self, reward_lst,q_ids):
        for i in range(len(reward_lst)):
            self._update_q_reward(reward_lst[i],q_ids[i])



    def _update_q_reward(self,r,q_id):

        cur_rewards = self.base_reward.get(q_id, [])

        if len(cur_rewards) < self.window_size:
            if r == 0:
                r = 0.0001
            cur_rewards.append(r)
        else:
            cur_rewards[self.iter] = r


        self.base_reward.update({q_id:cur_rewards})