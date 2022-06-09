import numpy as np

### Help classes
class Cum_mean:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim)
    def __call__(self, x):
        self.mean = (x + self.n * self.mean) / (self.n + 1)
        self.n += 1
        return self.mean
            
class Cum_std:
    def __init__(self, dim):
        self.sum1 = np.zeros(dim)
        self.sum2 = np.zeros(dim)
        self.n = 0
        self.std = np.ones(dim)
    def __call__(self, x):
        self.sum1 += x
        self.sum2 += x*x
        self.n    += 1.0
        sum1, sum2, n = self.sum1, self.sum2, self.n
        if self.n == 0:
            self.std = np.zeros_like(x)
        else:
            self.std = np.sqrt(self.sum2/self.n - self.sum1*self.sum1/self.n/self.n) 
        return self.std
    
#########################################################################
### Linear models
class Linear_model:
    def __init__(self, n_state, n_action): 
        # M stands for parameters of this model
        self.M = np.zeros((n_action, n_state))
        
    def __call__(self, state, noise=0):
        action = (self.M + noise) @ state
        return action
    
class Linear_model_v2:
    def __init__(self, n_state, n_action): 
        # M stands for parameters of this model
        self.M = np.zeros((n_action, n_state))
        
        self.cum_mu = Cum_mean(n_state)
        self.cum_sigma = Cum_std(n_state)
        
    def __call__(self, state, noise=0):
        # [(n_action, n_state) + (n_action, n_state)] @ (n_state,n_state) @ [(n_state, 1) - (n_state, 1)]
        _sigma = np.diag(self.cum_sigma.std)
        action = (self.M + noise) @ _sigma @ (state - self.cum_mu.mean)
        return action
    
#########################################################################
### Optimaizers
'''
BRS_v1    (V1 model with simple update)
'''
class BaseRandomSearch:
    def __init__(self, n_state, n_action, std):
        self.policy = Linear_model(n_state,n_action)
        self.std = std
        
        self.buffer = []
        
    def get_actions(self, state, is_train=False):
        if is_train:
            noise = np.random.normal(0, self.std, size=self.policy.M.shape)
            action_pos = self.policy(state,  noise)
            action_neg = self.policy(state, -noise)
            action_pos = self._remap_actions(action_pos)
            action_neg = self._remap_actions(action_neg)
            return action_pos, action_neg, noise
        else:
            action = self.policy(state)
            return self._remap_actions(action)
        
    def learn(self):
        update = np.zeros_like(self.policy.M)
        for step in self.buffer:
            update += ((step[-2] - step[-1]) * step[-3]) / self.std
        
        self.policy.M = self.policy.M + update  
        
        self.buffer = []
        
    def remember(self, memory):
        self.buffer.append(memory)
        
    def _remap_actions(self, action):
        return np.tanh(action)
    
'''
BRS_v1_RS (V1 model with random subset buffer update)
'''
class BRS_RS(BaseRandomSearch):
    def __init__(self, n_state, n_action, std, n_samples):
        self.policy = Linear_model(n_state,n_action)
        self.std = std
        self.n_samples = n_samples
        
        self.buffer = []
        
    def learn(self):
        update = np.zeros_like(self.policy.M)
        
        buffer_idxs = np.random.choice(np.arange(len(self.buffer)), self.n_samples)
        buffer = np.array(self.buffer, dtype=object)[buffer_idxs]
        
        for step in buffer:
            update += ((step[-2] - step[-1]) * step[-3]) / self.std
        
        self.policy.M = self.policy.M + update  
        
        self.buffer = []
        
'''
BRS_v1_SS (V1 model with sorted subset buffer update)
'''
class BRS_SS(BRS_RS):
    def _sort_directions(self):
        buffer = np.array(self.buffer, dtype=object)
        b_rewards = np.max(buffer[:, -2:], 1)

        # idxs from low to high
        b_idxs = np.argsort(b_rewards)
        b_buffer = buffer[b_idxs][-self.n_samples:]

        return b_buffer
    

    def learn(self):
        update = np.zeros_like(self.policy.M)
        
        buffer = self._sort_directions()
        
        for step in buffer:
            update += ((step[-2] - step[-1]) * step[-3]) / self.std
        
        self.policy.M = self.policy.M + update  
        
        self.buffer = []
        
'''
BRS_v1_SSn (Normalize state before training)
'''
class BRS_SSn(BRS_SS):
    def __init__(self, n_state, n_action, std, n_samples, state_low, state_high):
        self.policy = Linear_model(n_state,n_action)
        self.std = std
        self.n_samples = n_samples
        self.state_low = state_low
        self.state_high = state_high
        
        self.buffer = []
    
    def get_actions(self, state, is_train=False):
        state = (state - self.state_low)/(self.state_high - self.state_low)
        if is_train:
            noise = np.random.normal(0, self.std, size=self.policy.M.shape)
            action_pos = self.policy(state,  noise)
            action_neg = self.policy(state, -noise)
            action_pos = self._remap_actions(action_pos)
            action_neg = self._remap_actions(action_neg)
            return action_pos, action_neg, noise
        else:
            action = self.policy(state)
            return self._remap_actions(action)
        
'''
BRS_v2    (V2 model with simple update)
'''
class BaseRandomSearch_v2:
    def __init__(self, n_state, n_action, std):
        self.policy = Linear_model_v2(n_state,n_action)
        self.std = std
        
        self.buffer = []
        self.states = []
        
    def get_actions(self, state, is_train=False):
        if is_train:
            noise = np.random.normal(0, self.std, size=self.policy.M.shape)
            action_pos = self.policy(state,  noise)
            action_neg = self.policy(state, -noise)
            action_pos = self._remap_actions(action_pos)
            action_neg = self._remap_actions(action_neg)
            return action_pos, action_neg, noise
        else:
            action = self.policy(state)
            return self._remap_actions(action)
        
    def learn(self):
        update = np.zeros_like(self.policy.M)
        for step in self.buffer:
            update += ((step[-2] - step[-1]) * step[-3]) / self.std
        
        self.policy.M = self.policy.M + update  
        
        for s in self.states:
            self.policy.cum_mu(s)
            self.policy.cum_sigma(s)
        # given by paper
        self.policy.cum_sigma.std[self.policy.cum_sigma.std<1e-8] = 1e+8
        
        self.buffer = []
        self.states = []
        
        
    def remember(self, memory):
        # pos_action, neg_action, noise, pos_reward, neg_reward
        self.buffer.append(memory)
        
    def _remap_actions(self, action):
        return np.tanh(action)
            
    
'''
BRS_v2_RS (V2 model with random subset buffer update)
'''
class BRS_v2_RS(BaseRandomSearch_v2):
    def __init__(self, n_state, n_action, std, n_samples):
        self.policy = Linear_model_v2(n_state,n_action)
        self.std = std
        self.n_samples = n_samples
        
        self.buffer = []
        self.states = []
        
    def learn(self):
        update = np.zeros_like(self.policy.M)
        
        buffer_idxs = np.random.choice(np.arange(len(self.buffer)), self.n_samples)
        buffer = np.array(self.buffer, dtype=object)[buffer_idxs]
        
        for step in buffer:
            update += ((step[-2] - step[-1]) * step[-3]) / self.std
        
        self.policy.M = self.policy.M + update 
        
        for s in self.states:
            self.policy.cum_mu(s)
            self.policy.cum_sigma(s)
        # given by paper
        self.policy.cum_sigma.std[self.policy.cum_sigma.std<1e-8] = 1e+8
        
        self.buffer = []
        self.states = []
    
    
'''
BRS_v2_SS (V2 model with sorted subset buffer update)
'''
class BRS_v2_SS(BRS_v2_RS):
    def _sort_directions(self):
        buffer = np.array(self.buffer, dtype=object)
        b_rewards = np.max(buffer[:, -2:], 1)

        # idxs from low to high
        b_idxs = np.argsort(b_rewards)
        b_buffer = buffer[b_idxs][-self.n_samples:]

        return b_buffer

    def learn(self):
        update = np.zeros_like(self.policy.M)
        
        buffer = self._sort_directions()
        
        for step in buffer:
            update += ((step[-2] - step[-1]) * step[-3]) / self.std
        
        self.policy.M = self.policy.M + update
        
        for s in self.states:
            self.policy.cum_mu(s)
            self.policy.cum_sigma(s)
        
        # given by paper
        self.policy.cum_sigma.std[self.policy.cum_sigma.std<1e-8] = 1e+8
        
        self.buffer = []
        self.states = []
        
    
    
'''
BRS_v2_SSn (Normalize state before training)
'''
class BRS_v2_SSn(BRS_v2_SS):
    def __init__(self, n_state, n_action, std, n_samples, state_high, state_low):
        self.policy = Linear_model_v2(n_state,n_action)
        self.std = std
        self.n_samples = n_samples
        self.state_high = state_high 
        self.state_low = state_low
        
        self.buffer = []
        self.states = []
        
    def get_actions(self, state, is_train=False):
        state = (state - self.state_low)/(self.state_high - self.state_low)
        if is_train:
            noise = np.random.normal(0, self.std, size=self.policy.M.shape)
            action_pos = self.policy(state,  noise)
            action_neg = self.policy(state, -noise)
            action_pos = self._remap_actions(action_pos)
            action_neg = self._remap_actions(action_neg)
            return action_pos, action_neg, noise
        else:
            action = self.policy(state)
            return self._remap_actions(action)
                
    
    
'''
ARS_v1
'''
class ARS_v1:
    def __init__(self, n_state, n_action, std, alpha):
        self.policy = Linear_model(n_state,n_action)
        self.std = std
        self.alpha = alpha
        
        self.buffer = []
        self.std_rewards_over_training = []
        
    def get_actions(self, state, is_train=False):
        if is_train:
            noise = np.random.normal(0, self.std, size=self.policy.M.shape)
            action_pos = self.policy(state,  noise)
            action_neg = self.policy(state, -noise)
            action_pos = self._remap_actions(action_pos)
            action_neg = self._remap_actions(action_neg)
            return action_pos, action_neg, noise
        else:
            action = self.policy(state)
            return self._remap_actions(action)

    def learn(self):
        n_samples = len(self.buffer)
        b_buffer = np.array(self.buffer)
        update = np.zeros_like(self.policy.M)
        
        
        for step in b_buffer:
            r_p = step[-2]
            r_n = step[-1]
            noise = step[-3]
            update += ((r_p - r_n) * noise)
        
        reward_std = b_buffer[:, -2:].std()
        norm = self.alpha / (n_samples * reward_std)
        
        self.policy.M = self.policy.M + (norm * update)
        
        self.buffer = []
        self.std_rewards_over_training += [reward_std]
        
    def remember(self, memory):
        # pos_action, neg_action, noise, pos_reward, neg_reward
        self.buffer.append(memory)
        
    def _remap_actions(self, action):
        return np.tanh(action)
            
    
'''
ARS_v1_RS
'''
class ARS_v1_RS(ARS_v1):
    def __init__(self, n_state, n_action, std, n_samples, alpha):
        self.policy = Linear_model(n_state,n_action)
        self.std = std
        self.alpha = alpha
        self.n_samples = n_samples
        
        self.buffer = []
        self.std_rewards_over_training = []
        

    def learn(self):
        buffer_idxs = np.random.choice(np.arange(len(self.buffer)), self.n_samples)
        b_buffer = np.array(self.buffer, dtype=object)[buffer_idxs]
        
        update = np.zeros_like(self.policy.M)
        for step in b_buffer:
            r_p = step[-2]
            r_n = step[-1]
            noise = step[-3]
            update += ((r_p - r_n) * noise)
        
        reward_std = b_buffer[:, -2:].std()
        norm = self.alpha / (self.n_samples * reward_std)
        
        self.policy.M = self.policy.M + (norm * update)
        
        self.buffer = []
        self.std_rewards_over_training += [reward_std]
          
    
    
'''
ARS_v1_SS
'''
class ARS_v1_SS(ARS_v1_RS):    
    def _sort_directions(self):
        buffer = np.array(self.buffer, dtype=object)
        b_rewards = buffer[:, -2:].sum(1)

        # idxs from low to high
        b_idxs = np.argsort(b_rewards)
        b_buffer = buffer[b_idxs][-self.n_samples:]

        return b_buffer

    def learn(self):
        b_buffer = self._sort_directions()
        update = np.zeros_like(self.policy.M)
        
        for step in b_buffer:
            r_p = step[-2]
            r_n = step[-1]
            noise = step[-3]
            update += ((r_p - r_n) * noise)
        
        reward_std = b_buffer[:, -2:].std()
        norm = self.alpha / (self.n_samples * reward_std)
        
        self.policy.M = self.policy.M + (norm * update)
        
        self.buffer = []
        self.std_rewards_over_training += [reward_std]
        
    
'''
ARS_v1_SSn
'''
class ARS_v1_SSn(ARS_v1_SS):
    def __init__(self, n_state, n_action, std, n_samples, alpha, state_low, state_high):
        self.policy = Linear_model(n_state,n_action)
        self.std = std
        self.alpha = alpha
        self.n_samples = n_samples
        
        self.buffer = []
        self.std_rewards_over_training = []
        
        self.state_low=state_low
        self.state_high=state_high
        
    def get_actions(self, state, is_train=False):
        if (self.state_high is not None and self.state_low is not None):
            # MINMAX NORM {value - min}/{max - min}
            state = (state - self.state_low)/(self.state_high - self.state_low)
    
        if is_train:
            noise = np.random.normal(0, self.std, size=self.policy.M.shape)
            action_pos = self.policy(state,  noise)
            action_neg = self.policy(state, -noise)
            action_pos = self._remap_actions(action_pos)
            action_neg = self._remap_actions(action_neg)
            return action_pos, action_neg, noise
        else:
            action = self.policy(state)
            return self._remap_actions(action)
    
    
'''
ARS_v2
'''
class ARS_v2(ARS_v1):
    def __init__(self, n_state, n_action, std, alpha):
        self.policy = Linear_model_v2(n_state,n_action)
        self.std = std
        self.alpha = alpha
        
        self.buffer = []
        self.states = []
        
        self.std_rewards_over_training = []

    def learn(self):
        n_samples = len(self.buffer)
        b_buffer = np.array(self.buffer)
        
        update = np.zeros_like(self.policy.M)
        for step in b_buffer:
            r_p = step[-2]
            r_n = step[-1]
            noise = step[-3]
            update += ((r_p - r_n) * noise)
        
        reward_std = b_buffer[:, -2:].std()
        norm = self.alpha / (n_samples * reward_std)
        
        self.policy.M = self.policy.M + (norm * update)
        
        for s in self.states:
            self.policy.cum_mu(s)
            self.policy.cum_sigma(s)
        
        # given by paper
        self.policy.cum_sigma.std[self.policy.cum_sigma.std<1e-8] = 1e+8
        
        self.buffer = []
        self.states = []
        
        self.std_rewards_over_training += [reward_std]
    
'''
ARS_v2_RS
'''
class ARS_v2_RS(ARS_v2):
    def __init__(self, n_state, n_action, std, alpha, n_samples):
        self.policy = Linear_model_v2(n_state,n_action)
        self.std = std
        self.alpha = alpha
        self.n_samples = n_samples
        
        self.buffer = []
        self.states = []
        
        self.std_rewards_over_training = []

    def learn(self):
        buffer_idxs = np.random.choice(np.arange(len(self.buffer)), self.n_samples)
        b_buffer = np.array(self.buffer, dtype=object)[buffer_idxs]
        
        update = np.zeros_like(self.policy.M)
        for step in b_buffer:
            r_p = step[-2]
            r_n = step[-1]
            noise = step[-3]
            update += ((r_p - r_n) * noise)
        
        reward_std = b_buffer[:, -2:].std()
        norm = self.alpha / (self.n_samples * reward_std)
        
        self.policy.M = self.policy.M + (norm * update)
        
        for s in self.states:
            self.policy.cum_mu(s)
            self.policy.cum_sigma(s)
        
        # given by paper
        self.policy.cum_sigma.std[self.policy.cum_sigma.std<1e-8] = 1e+8
        
        self.buffer = []
        self.states = []
        
        self.std_rewards_over_training += [reward_std]
    
'''
ARS_v2_SS
'''
class ARS_v2_SS(ARS_v2_RS):
    def _sort_directions(self):
        buffer = np.array(self.buffer, dtype=object)
        b_rewards = buffer[:, -2:].sum(1)

        # idxs from low to high
        b_idxs = np.argsort(b_rewards)
        b_buffer = buffer[b_idxs][-self.n_samples:]

        return b_buffer

    
    def learn(self):
        b_buffer = self._sort_directions()
        
        update = np.zeros_like(self.policy.M)
        for step in b_buffer:
            r_p = step[-2]
            r_n = step[-1]
            noise = step[-3]
            update += ((r_p - r_n) * noise)
        
        reward_std = b_buffer[:, -2:].std()
        norm = self.alpha / (self.n_samples * reward_std)
        
        self.policy.M = self.policy.M + (norm * update)
        
        for s in self.states:
            self.policy.cum_mu(s)
            self.policy.cum_sigma(s)
        
        # given by paper
        self.policy.cum_sigma.std[self.policy.cum_sigma.std<1e-8] = 1e+8
        
        self.buffer = []
        self.states = []
        
        self.std_rewards_over_training += [reward_std]

'''
ARS_v2_SSn
'''
class ARS_v2_SSn(ARS_v2_SS):
    def __init__(self, n_state, n_action, std, n_samples, alpha, state_low, state_high):
        self.policy = Linear_model_v2(n_state,n_action)
        self.std = std
        self.alpha = alpha
        self.n_samples = n_samples
        
        self.state_low=state_low
        self.state_high=state_high
        
        self.buffer = []
        self.states = []
        
        self.std_rewards_over_training = []
        
    def get_actions(self, state, is_train=False):
        if (self.state_high is not None and self.state_low is not None):
            # MINMAX NORM {value - min}/{max - min}
            state = (state - self.state_low)/(self.state_high - self.state_low)
    
        if is_train:
            noise = np.random.normal(0, self.std, size=self.policy.M.shape)
            action_pos = self.policy(state,  noise)
            action_neg = self.policy(state, -noise)
            action_pos = self._remap_actions(action_pos)
            action_neg = self._remap_actions(action_neg)
            return action_pos, action_neg, noise
        else:
            action = self.policy(state)
            return self._remap_actions(action)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            