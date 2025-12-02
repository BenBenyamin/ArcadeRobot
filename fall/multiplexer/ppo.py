import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
import gymnasium as gym

class ActorCritic(nn.Module):
    ## Adapted from cleanRL , here : https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
    def __init__(self,state_dim, action_dim, hidden_size=64 ,actor = None, continuous=False):
        super().__init__()

        self.continuous = continuous

        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        if actor is None:
        # Shared or separate networks
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, action_dim)
            )
        else:
            self.actor = actor

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self, actor_std=0.01, critic_std=1.0, bias_const=0.0):
        """Applies orthogonal initialization to all linear layers."""

        actor_linear_layers = [layer for layer in self.actor.modules() if isinstance(layer, nn.Linear)]
        # Actor layers
        for i, layer in enumerate(actor_linear_layers):
            if isinstance(layer, nn.Linear):
                std = actor_std if i == len(actor_linear_layers)-1 else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=std)
                nn.init.constant_(layer.bias, bias_const)

        # Critic layers
        for i, layer in enumerate(self.critic):
            if isinstance(layer, nn.Linear):
                std = critic_std if i == len(self.critic)-1 else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=std)
                nn.init.constant_(layer.bias, bias_const)
    
    def forward(self,x):

        value = self.critic(x)

        action_dist  = self.actor(x)

        if self.continuous:
            std = torch.exp(self.log_std).clamp(1e-6, 2.0)
            action_dist = D.Independent(D.Normal(action_dist, std), 1)
        else:
            action_dist = D.Categorical(logits=action_dist)

        return action_dist, value


class ClipSurrogatedObjectiveLoss(nn.Module):
    
    def __init__(self, eps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = eps
        
    def forward(self, ratio, adv):

        return -1.0*torch.min(
            ratio*adv,
            torch.clamp(ratio,1-self.eps,1+self.eps)*adv
        ).mean()

class ValueFunctionLoss(nn.Module):

    def __init__(self, coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeff = coeff

    def forward(self,Rt,V):

        return self.coeff * 0.5 * torch.pow(V - Rt,2).mean()

class EntropyBonus(nn.Module):

    def __init__(self, coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeff = coeff

    def forward(self,action_dist):

        return -self.coeff * action_dist.entropy().mean()


class PPOLoss(nn.Module):

    def __init__(self, eps, value_c,entropy_c, kl_coeff, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clip_loss = ClipSurrogatedObjectiveLoss(eps)
        self.value_loss = ValueFunctionLoss(value_c)
        self.ent_bonus = EntropyBonus(entropy_c)

        self.kl_coeff = kl_coeff
        
    
    def forward(self ,adv , Rt,V,actions, actions_dist , old_logprob):

        new_log_prob = actions_dist.log_prob(actions)
        log_ratio = new_log_prob - old_logprob
        ratio = torch.exp(log_ratio)

        kl = ((log_ratio.exp() - 1) - log_ratio).mean()
        clip_loss = self.clip_loss(ratio,adv)
        value_loss = self.value_loss(Rt,V)
        ent_bonus = self.ent_bonus(actions_dist)
        loss = clip_loss + value_loss + ent_bonus + self.kl_coeff * kl

        return  loss, kl

class GeneralizedAdvantageEstimation(nn.Module):
    def __init__(self, gamma=0.99, lam=0.95):
        super().__init__()
        self.gamma = gamma
        self.lam = lam

    def forward(self, rewards, values, dones, norm=True):
        """
        rewards: [T]
        values:  [T+1]  (bootstrap value for last state included)
        dones:   [T]
        """

        if values.dim() == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)

        if len(rewards.shape) == 1:
            T = rewards.shape[0]
            N = 1
        else:
            T, N = rewards.shape

        advantages = torch.zeros((T, N), device=rewards.device, dtype=rewards.dtype)
        last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * values[t + 1] * next_nonterminal - values[t]
            advantages[t]  = delta + self.gamma * self.lam * next_nonterminal * last_gae
            last_gae = advantages[t]
    


        returns = advantages + values[:-1]

        if norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        return advantages, returns

class PPOTrainer:
    def __init__(
        self,
        policy: ActorCritic,
        env,
        num_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        norm_adv: bool = True,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        kl_coeff: float = 0.02,
        max_grad_norm: float = 0.5,
        target_kl: float = None,
        learning_rate: float = 2.5e-4,
        anneal_lr: bool = True,
    ):
        
        self.policy = policy
        self.env = env
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vec = hasattr(env, "num_envs")
        self.obs_space = env.observation_space
        self.action_space = env.action_space

        self.device = policy.actor[0].weight.device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_steps = num_steps

        # GAE module
        self.gae = GeneralizedAdvantageEstimation(gamma, gae_lambda)

        # PPO loss
        self.loss_func = PPOLoss(
            eps=clip_coef,
            value_c=vf_coef,
            entropy_c=ent_coef,
            kl_coeff=kl_coeff,
        )

        # Other hyperparameters
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr

        self.num_timesteps = 0

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,
            
        )

        self.anneal_lr = anneal_lr
            
    def rollout(
            self,
        ):

        obs = self.env.reset()
        if not self.is_vec:
            obs = obs[0]

        rewards , values, dones , actions_list , obs_list , old_logprob = [], [], [] , [] , [] , []

        for _ in range(self.num_steps):
            
            obs_t = torch.as_tensor(obs, dtype=torch.float32 , device= self.device)
            obs_list.append(obs_t)

            actions_dist, value  = self.policy(obs_t)
            actions = actions_dist.sample()

            if self.is_vec:
                # SB3 VecEnv: step() → obs, rewards, dones, infos
                obs , reward, done_vec, infos = self.env.step(actions.cpu().numpy())

                # Extract truncated from info
                truncated_vec = torch.as_tensor(
                    [info.get("TimeLimit.truncated", False) for info in infos],
                    dtype=torch.bool,
                    device=self.device,
                )
                terminated_vec = torch.as_tensor(done_vec, dtype=torch.bool, device=self.device)
                done_flags = terminated_vec | truncated_vec
                
            else:

                obs, reward, terminated, truncated , _ = self.env.step(actions.cpu().numpy())
                done_flags = terminated or truncated

                if done_flags:
                    obs , _ = self.env.reset()
            
            done_flags = torch.tensor(done_flags, dtype= torch.bool, device = self.device)
            
            rewards.append(torch.as_tensor(reward, dtype=torch.float32 , device=self.device))
            values.append(value.detach())
            dones.append(done_flags)
            actions_list.append(actions.detach())
            old_logprob.append(actions_dist.log_prob(actions).detach())

            self.num_timesteps +=self.num_envs
   
        
        # Add value of the last observation (s_T)
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32 , device=self.device)
            _, last_value = self.policy(obs_t)
        values.append(last_value.detach())

        rewards = torch.stack(rewards).to(self.device)
        values = torch.stack(values).to(self.device)
        dones = torch.stack(dones).to(torch.bool).to(self.device)
        obs_tensor = torch.stack(obs_list).to(self.device)
        actions_tensor = torch.stack(actions_list).to(self.device)
        old_logprob = torch.stack(old_logprob).to(self.device)

        return rewards , values, dones, obs_tensor ,actions_tensor , old_logprob
                
    def update(self , rewards , values, dones, obs_tensor ,actions_tensor , old_logprob):

        advantages, returns = self.gae(rewards,values,dones,norm = self.norm_adv)

        batch_size = self.num_steps * self.num_envs
        minibatch_size = batch_size // self.num_minibatches
        indices = torch.randperm(batch_size, device=self.device)

        # flatten once per update
        obs_flat = obs_tensor.reshape(batch_size, -1)

        if self.policy.continuous:
            actions_flat = actions_tensor.reshape(batch_size, -1)
        else:
            actions_flat = actions_tensor.reshape(batch_size) 
        old_logprob_flat = old_logprob.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size, -1)

        for start in range(0, batch_size, minibatch_size):

            # Get minibatch
            mb_inds = indices[start:start+minibatch_size]

            # select minibatch 
            mb_obs = obs_flat[mb_inds].to(self.device)
            mb_actions = actions_flat[mb_inds].to(self.device)
            mb_old_logprob = old_logprob_flat[mb_inds].to(self.device)
            mb_advantages = advantages_flat[mb_inds].to(self.device)
            mb_returns = returns_flat[mb_inds].to(self.device)
            
            new_actions_dist , new_values = self.policy(mb_obs)
            
            loss , kl = self.loss_func(
                adv=mb_advantages,
                Rt=mb_returns,
                V=new_values,
                actions=mb_actions,
                actions_dist=new_actions_dist,
                old_logprob = mb_old_logprob,
            )

            if self.target_kl and kl > self.target_kl:
                break

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

    
    def train(self, num_timesteps:int):

        while self.num_timesteps < num_timesteps:

            if self.anneal_lr:
                frac = 1 - (self.num_timesteps / num_timesteps)
                lr_now = self.learning_rate * frac
                for g in self.optimizer.param_groups:
                    g["lr"] = lr_now

            rewards , values, dones, obs_tensor ,actions_tensor , old_logprob = self.rollout()

            for _ in range(self.update_epochs):

                self.update(rewards , values, dones, obs_tensor ,actions_tensor , old_logprob)

class PPO:

    def __init__(
        self,
        env,
        device = "cpu",
        actor = None,
        hidden_size=64,
        num_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=4,
        norm_adv=True,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        kl_coeff=0.02,
        max_grad_norm=0.5,
        target_kl=None,
        learning_rate=2.5e-4,
        anneal_lr=True,
    ):
        
        self.env = env

        obs_dim  = env.observation_space.shape[0]

        self.device = device

        if isinstance(env.action_space, gym.spaces.Discrete):
            continuous = False
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            continuous = True
            action_dim = env.action_space.shape[0]
        else:
            raise ValueError("Unsupported action space type:", env.action_space)

        action_dim = int(action_dim)

        self.policy = ActorCritic(
            actor=actor,
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            continuous=continuous
        ).to(self.device)

        self.trainer = PPOTrainer(
            policy=self.policy,
            env=self.env,
            num_steps=num_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            num_minibatches=num_minibatches,
            update_epochs=update_epochs,
            norm_adv=norm_adv,
            clip_coef=clip_coef,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            kl_coeff=kl_coeff,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            learning_rate=learning_rate,
            anneal_lr=anneal_lr,
        )

    def learn(self, total_timesteps):

        self.trainer.train(total_timesteps)
        return self

    def predict(self, obs, deterministic=True):

        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device)

        with torch.inference_mode():
            dist, _ = self.policy(obs)

            if self.policy.continuous:
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()
            else:
                if deterministic:
                    action = torch.argmax(dist.probs)
                else:
                    action = dist.sample()

        return action.cpu().numpy()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
