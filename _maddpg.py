"""
-> MADDPG  with action noise
-> Implemented the paper: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" by Lowe et al.
"""
import os
import numpy as np
import torch
import torch as T 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
# from transformer import *


SEED = 13
"""/////////// Replay Buffer /////////////"""
class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        """ critic_dims: (flattened) state vector, size =(obs_len*n_agents) """
        self.mem_size = max_size    # buffer size
        self.mem_cntr = 0           # buffer counter
        self.n_agents = n_agents
        self.actor_dims = actor_dims  # list of each agent's obs dim, size =(n_agents,)
        self.batch_size = batch_size
        self.n_actions = n_actions    
        # number of action dims for each agent (each action dim is continuous)
        self.state_memory = np.zeros((self.mem_size, critic_dims))  # flattened state vector
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool) 
        # terminal states have zero Q-values
        self.init_actor_memory()  # init actor memory, list of arrays

        # self.actor_state_memory = []  # list of (obs_len, ) arrays
        # self.actor_new_state_memory = []
        # self.actor_action_memory = []
        
    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        """ init buffer for each agent """
        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))  # obs input to actor i
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))
            
    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size
        """ raw_obs: list of local obs vectors, size =(n_agents, obs_len)
            state: flattened state vector
            action: vec of agent actions, size =(n_agent,) """ 
        for agent_idx in range(self.n_agents):  # store trans for actors
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]
        self.state_memory[index] = state     # store trans for critic  
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward       # reward: 1D array 
        self.terminal_memory[index] = done 
        self.mem_cntr += 1
        
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)  #returns 1D array  XY: Not a number?
        """for critic use"""
        states = self.state_memory[batch]        #np array, size =(batch_size,critic_dims)
        rewards = self.reward_memory[batch]      #size = (batch_size, n_agents)
        states_ = self.new_state_memory[batch] 
        terminal = self.terminal_memory[batch]   #size = (batch_size, n_agents)  XY: What is terminal and when to use it?
        """for actor use"""
        actor_states = []  #list of arrays (size=(batch_size,self.actor_dims[i])) for agent i
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
            
        return actor_states, states, actions, rewards, actor_new_states, states_, terminal 
    
    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True 
        return False   #do not learn if samples less than a batch


"""////Networks: actor, critic, target actor, target critic /// """
class CriticNetwork(nn.Module):
    """ NN input: state, joint action;
        NN output: q(state, joint action)
    """
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_agents, 
                 n_actions, name, chkpt_dir):   
        """INPUT:
            input_dims: list-like [actor_dims*n_agents]
        """
        super(CriticNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)  #join one or more path components

        self.fc1 = nn.Linear(in_features=input_dims+n_agents*n_actions, out_features=fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        # self.ln1 = nn.BatchNorm1d(fc1_dims)

        self.fc2 = nn.Linear(in_features=fc1_dims, out_features=fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        # self.ln2 = nn.BatchNorm1d(fc2_dims)

        self.fc3 = nn.Linear(in_features=fc2_dims, out_features=fc3_dims)
        self.ln3 = nn.LayerNorm(fc3_dims)
        # self.ln3 = nn.BatchNorm1d(fc3_dims)

        self.q = nn.Linear(in_features=fc3_dims, out_features=1)
        """init NN weights"""
        T.manual_seed(SEED)
        T.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")  #or uniform
        T.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        T.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
        T.nn.init.kaiming_normal_(self.q.weight, nonlinearity="relu")
        """optimizer"""
        self.optimizer = optim.Adam(self.parameters(), lr=beta)   #lr is fixed  XY: Beta is same as gamma?
        # self.optimizer = optim.SGD(params=self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)       
        
    def forward(self, state, action):
        """INPUT:
            state: size =(batch_size, state_vec_len)
            action: size =(batch_size, joint_action_len)
        """
        x = F.relu(self.ln1(self.fc1(T.cat([state, action], dim=1))))  # concat along dim 1
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        #
        # x = F.relu(self.fc1(T.cat([state, action], dim=1)))  #concat along dim 1
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))

        q = F.relu(self.q(x))    #2D-like, size=(batch_size,1),relu used cuz Q>=0
        return q
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)     # save model
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))    # load model
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, 
                 name, chkpt_dir):
        """INPUT:
            n_actions: number of action dims (each being continuous) of each agent
        """
        super(ActorNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        # print(fc1_dims, fc2_dims, fc3_dims)
        self.fc1 = nn.Linear(in_features=input_dims, out_features=fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        # self.ln1 = nn.BatchNorm1d(fc1_dims)

        self.fc2 = nn.Linear(in_features=fc1_dims, out_features=fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        # self.ln2 = nn.BatchNorm1d(fc2_dims)

        self.fc3 = nn.Linear(in_features=fc2_dims, out_features=fc3_dims)
        self.ln3 = nn.LayerNorm(fc3_dims)
        # self.ln3 = nn.BatchNorm1d(fc3_dims)

        self.pi = nn.Linear(in_features=fc3_dims, out_features=n_actions)
        """init NN weights"""
        T.manual_seed(SEED)
        T.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        T.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        T.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
        """optimizer """
        self.optimizer = optim.Adam(params=self.parameters(), lr=alpha)
        # self.optimizer = optim.SGD(params=self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)       # push model to device 
        
    def forward(self, state):
        """INPUT/OUTPUT:
            state: batched states, size=(batch_size, len(state_vector))
            pi: actions (to be clipped), size=(batch_size, n_actions) 
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        """ softmax/sigmoid/tanh to constrain action vals"""
        # pi = T.sigmoid(input=self.pi(x))     #!!!sigmoid activation [0, 1]
        pi = T.tanh(input=self.pi(x))    #!!! tanh activation as output [-1, 1]
        # print(pi)
        # print(pi.shape)

        return pi 
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)     # save model
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))    # load model


""" //////////// Exploration noise //////////////"""
class UniformNoise(object):
    def __init__(self, width, width_min, width_decay, decay_freq, size):
        """INPUT:
            width: rv in [-width, width]
            width_min: min width [-a_min, a_min]
            width_decay: decay width by this factor each time
            decay_freq: decay width every this func call 
            size: noise vector len
        """
        self.width = width
        self.width_min = width_min 
        self.width_decay = width_decay
        self.decay_freq = decay_freq
        self.size = size
        self.cntr = 0
        
    def __call__(self):
        if self.cntr % self.decay_freq == 0:
            self.width = max(self.width_min, self.width * self.width_decay)
        out = 2 * self.width * (T.rand(size=tuple([self.size])) - 0.5)  #scale to [-1,1]
        self.cntr += 1
        return out

class GaussianNoise(object):
    def __init__(self, mean, std, std_min, std_decay, decay_freq, size):
        """INPUT:
            mean: scalar
            std: scalar
            std_min: minimum std
            std_decay: decay std by this each time
            decay_freq: decay std every this func call
            size: noise vector len
        """
        self.mean = mean
        self.std = std
        self.std_min = std_min
        self.std_decay = std_decay
        self.decay_freq = decay_freq
        self.size = size
        self.cntr = 0
    
    def __call__(self):
        if self.cntr % self.decay_freq == 0:
            self.std = max(self.std_min, self.std * self.std_decay)
        out = T.normal(mean=T.tensor(self.mean*np.ones(self.size), dtype=T.float),
                     std=T.tensor(self.std*np.ones(self.size), dtype=T.float))  #tensors returned
        self.cntr += 1
        return out


"""--------------------------------------MADDPG-------------------------------------------"""
"""/////////// Agent /////////////"""
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                 alpha=0.01, beta=0.01, fc1=64, fc2=64, fc3=64, gamma=0.9, tau=0.001,
                 action_noise="Gaussian", noise_init=1.0, noise_min=0.05,
                 noise_decay=1-7e-5):
        """INPUT:
            action_min: "Gaussian" or "Uniform", exploration noise. Type:string
            noise_min: min noise range (std for Gaussian, range for uniform)
            noise_decay: decay rate of exploration noise
        """
        self.gamma = gamma  # XY: discount factor
        self.tau = tau 
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        """main networks"""
        self.actor = ActorNetwork(alpha=alpha,
                                  input_dims=actor_dims,
                                  fc1_dims=fc1,
                                  fc2_dims=fc2,
                                  fc3_dims=fc3,
                                  n_actions=n_actions,
                                  name=self.agent_name + "_actor",
                                  chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(beta=beta,
                                    input_dims=critic_dims,
                                    fc1_dims=fc1,
                                    fc2_dims=fc2,
                                    fc3_dims=fc3,
                                    n_actions=n_actions,
                                    n_agents=n_agents,
                                    name=self.agent_name + "_critic",
                                    chkpt_dir=chkpt_dir)
        """ target networks"""
        self.target_actor = ActorNetwork(alpha=alpha,
                                         input_dims=actor_dims,
                                         fc1_dims=fc1,
                                         fc2_dims=fc2,
                                         fc3_dims=fc3,
                                         n_actions=n_actions,
                                         name=self.agent_name + "_target_actor",
                                         chkpt_dir=chkpt_dir)
        self.target_critic = CriticNetwork(beta=beta,
                                           input_dims=critic_dims,
                                           fc1_dims=fc1,
                                           fc2_dims=fc2,
                                           fc3_dims=fc3,
                                           n_actions=n_actions,
                                           n_agents=n_agents,
                                           name=self.agent_name + "_target_critic",
                                           chkpt_dir=chkpt_dir)
        """ exploration noise """
        if action_noise == "Gaussian":   
            self.noise = GaussianNoise(mean=0, std=noise_init, std_min=noise_min,
                            std_decay=noise_decay, decay_freq=1, size=n_actions)  #Gaussian noise
        else:
            self.noise = UniformNoise(width=noise_init, width_min=noise_min,
                            width_decay=noise_decay, decay_freq=1, size=n_actions)  #uniform noise
        self.update_network_parameters(tau=1)  # XY: Initial the parameters for all agents
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau 
        """ soft update actor network"""
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()
        
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1.0 - tau) * target_actor_state_dict[name].clone()     #clone then load back
        self.target_actor.load_state_dict(actor_state_dict)   #load updated params to target net
        """ soft update of critic network"""       
        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()
        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                (1.0 - tau) * target_critic_state_dict[name].clone()   
        self.target_critic.load_state_dict(critic_state_dict)   #load updated params to target net
        
    def choose_action(self, observation):
        """ add noise for exploration """
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)  # [obs]->add batch dim
        action = self.actor.forward(state)
        noise = self.noise().to(self.actor.device)
        # print(action, noise)

        # action = T.clamp(action + noise, min=0, max=1) #!!! clip noisy action, sigmoid actor output
        action = T.clamp(action + noise, min=-0.95, max=0.95)  #!!! clip noisy action, tanh actor output
        """NOTE: tanh output must be clipped to a smaller range [-0.95,0.95] as the
        extreme values -1,1 requires infinity input to achieve"""
        
        return action.detach().cpu().numpy()[0]   #remove output batch dim
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


"""///////////////// This class handles all agents ////////////////"""
class MADDPG:   
    """ define the wrapper class """
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, alpha=0.01, beta=0.01,
                 fc1=64, fc2=64, fc3=64, gamma=0.9, tau=0.001, action_noise="Gaussian",
                 noise_init=1.0, noise_min=0.05, noise_decay=1-7e-5, chkpt_dir="tmp/maddpg/"):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims=actor_dims[agent_idx], 
                                     critic_dims=critic_dims, 
                                     n_actions=n_actions,
                                     n_agents=n_agents,
                                     agent_idx=agent_idx,
                                     chkpt_dir=chkpt_dir,
                                     alpha=alpha,
                                     beta=beta,
                                     fc1=fc1,
                                     fc2=fc2,
                                     fc3=fc3,
                                     gamma=gamma,
                                     tau=tau,
                                     action_noise=action_noise,
                                     noise_init=noise_init,
                                     noise_min=noise_min,
                                     noise_decay=noise_decay))

    def save_checkpoint(self):
        print('... Saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()
            
    def load_checkpoint(self):
        print('... Loading checkpoint ...')
        for idx, agent in enumerate(self.agents):
            agent.load_models()
            print(f"Agent {idx:d} model loaded...")
            
    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions   #return a list of arrays
    
    def learn(self, memory, slot_idx, target_freq):
        """ define learn process"""
        if not memory.ready():
            return    # do not learn if number of samples < batch_size
        actor_states, states, actions, rewards, actor_new_states, states_, dones = \
            memory.sample_buffer()    # sample minibatch: [obs_list, state, actions, rwd_list, obs_list_, state_, done]s
        device = self.agents[0].actor.device 
        states = T.tensor(states, dtype=T.float).to(device)   # convert to tensors
        actions = T.tensor(np.array(actions), dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)
        
        all_agents_new_actions = []       # new actions used in the TD targ next state Q-val (line 11)
        all_agents_new_mu_actions = []    # actions a_i generated by actor mu_i Q(s)(paper Alg 1 line 14)
        old_agents_actions = []           # current actions in Q^mu (line 12)

        # print(slot_idx, 'actions: ', actions.shape)
        
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            # print(slot_idx, agent_idx, new_states.shape)
            new_pi = agent.target_actor.forward(new_states)
            # print(slot_idx, agent_idx, new_pi.shape)
            all_agents_new_actions.append(new_pi) 
            
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)  
            pi = agent.actor.forward(mu_states)    #a_i=mu_i(obs_i), for autograd of Q_i wrt mu_i  XY: ?
            all_agents_new_mu_actions.append(pi)    
            
            old_agents_actions.append(actions[agent_idx])
            
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1) 
        # size =(batch_size, n_actions * n_agents)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)  
        # size =(batch_size, n_actions * n_agents)
        agent.target_critic

        Actor_loss = []
        Critic_loss = []
        for agent_idx, agent in enumerate(self.agents):    
            """ Critic learn """
            # print(states_.shape, new_actions)
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()  # TD-target for target network
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()  #Q to be regressed

            target = rewards[:, agent_idx] + agent.gamma * critic_value_   #TD-target
            critic_loss = F.mse_loss(target, critic_value)
            # print(slot_idx, agent_idx, 'critic loss: ', critic_loss)
            agent.critic.optimizer.zero_grad()  
            critic_loss.backward(inputs=list(agent.critic.parameters()), retain_graph=True)
            agent.critic.optimizer.step()  #optimize critic for one step

            """ Actor learn """  # XY: A little confused, need to be better understood
            agent_pg_Q_input_actions = old_agents_actions  #(a_1^j,,a_i,,a_N^j) in Q_i [Alg 1 line 14]
            agent_pg_Q_input_actions[agent_idx] = all_agents_new_mu_actions[agent_idx]  #a_i=mu_i(obs_i)
            flat_actions = T.cat([acts for acts in agent_pg_Q_input_actions], dim=1)  #flatten acts 
            actor_loss = - T.mean(agent.critic.forward(states, flat_actions).flatten())
            # print(slot_idx, agent_idx, 'actor loss: ', actor_loss)
            #PG=E[gradient of Q wrt policy params]
            # print(critic_loss, actor_loss)
            agent.actor.optimizer.zero_grad()
            
            actor_loss.backward(inputs=list(agent.actor.parameters()), retain_graph=True)  #!
            # if torch.isinf(critic_loss):
            #     raise Exception("Infinite loss detected!")
            # print(agent_idx, critic_loss, actor_loss)
            
            # actor_loss.backward(inputs=list(agent.actor.parameters())) #
            Actor_loss.append(actor_loss)
            Critic_loss.append(critic_loss)
            
            agent.actor.optimizer.step()

            if slot_idx % target_freq == 0:
                agent.update_network_parameters()  #input tau is default to None, so Agent.tau is used
        # print(slot_idx, 'Actor loss: ', Actor_loss)
        # print(slot_idx, 'Critic loss: ', Critic_loss, '\n')
