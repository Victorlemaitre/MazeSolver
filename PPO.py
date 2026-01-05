import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor,device
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

class PPO_agent(nn.Module):
    def __init__(
            self, *,
            maze_size : int,
            nb_channel : int,
            device : torch.device = device,
            learning_rate : float = 1e-3,
            batch_size : int = 256,
            gamma : float = 0.99,
            gae_lambda : float = 0.95,
            eps : float = 0.1,
            n_pass : int = 5,
            target_KL : float = 0.01,
            entropy_coeff : float = 0.01,
            critic_coeff : float = 1
                 ):

        """
        gamma : the discount factor
        gae_lambda : the lambda parameter in GAE (Generalized Advantage Estimation)
        eps : the epsilon value in the clipped actor loss of PPO and also the clipped value loss
        n_pass : the epoch parameter in PPO's SGD on the collected trajectories
        target_KL : we exit the update early if the new policy KL's divergence with the reference gets too high
        entropy_coeff : controls the entropy loss's weight in the total loss
        critic_coeff : controls the critic's loss's weight in the total loss
        """
        super().__init__()
        assert maze_size > 5
        assert nb_channel > 0
        assert learning_rate > 0
        assert 0 <= gamma < 1
        assert 0 <= gae_lambda <= 1
        assert entropy_coeff >= 0
        assert critic_coeff >= 0
        assert n_pass > 0
        assert target_KL > 0
        assert batch_size > 0

        self.device = device
        self.maze_size = maze_size
        self.nb_channel = nb_channel

        final_hidden_size = 256

        self.backbone = self.backbone = nn.Sequential(
            nn.Conv2d(self.nb_channel, 32, 3, padding=0), #padding 0 cause the image is already padded
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((maze_size-2)*(maze_size-2)*128, final_hidden_size),
            nn.ReLU(),
            nn.Linear(final_hidden_size, final_hidden_size),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(final_hidden_size,4)
        )

        self.critic = nn.Sequential(
            nn.Linear(final_hidden_size,1)
        )

        self.optimizer = AdamW(self.parameters(), lr = learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5)


        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.eps = eps
        self.critic_coeff = critic_coeff
        self.n_pass = n_pass
        self.target_KL = target_KL
        self.batch_size = batch_size

        self.to(device)

    def get_value(self, observations : Tensor) -> Tensor:
        features = self.backbone(observations)
        value = self.critic(features).squeeze(-1)

        return value



    def get_action_value(self, observations : Tensor) -> tuple[Tensor, Tensor, Tensor]:

        features = self.backbone(observations)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.long().detach(), value, log_prob

    def get_log_prob_value_entropy(self, observations : Tensor, actions : Tensor):

        features = self.backbone(observations)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)

        log_probs = dist.log_prob(actions)  
        entropy = dist.entropy()
        values = self.critic(features).squeeze(-1)

        return log_probs, values, entropy



    def update(self,observations : Tensor, actions : Tensor, values : Tensor, log_probs : Tensor, advantage : Tensor, masks_done : Tensor) -> None:

        # the following valid idx are there to mask the actions taken on final observations. 
        # Indeed we only reset the env the next step in order to get a final value on the final observation for boostrapping when the trajectory is truncated
        filter_true_action = torch.ones_like(masks_done)
        filter_true_action[1:] = masks_done[:-1]
        valid = filter_true_action.bool()


        values = values[:-1]
        returns = advantage + values
        advantage = (advantage - advantage[valid].mean()) / (advantage[valid].std() + 1e-8)

        

        valid = valid.flatten()
        advantage = advantage.flatten()
        returns = returns.flatten().detach()
        observations = observations.reshape((-1, self.nb_channel, self.maze_size, self.maze_size)).contiguous()
        actions = actions.flatten()
        values = values.flatten()
        log_probs = log_probs.flatten()

        T = advantage.shape[0]
        nb_batches = T//self.batch_size
        for _ in range(self.n_pass):
            idx = torch.randperm(T)
            approx_kl_sum = 0
            kl_count = 0
            for i in range(0,T,self.batch_size): 
                if i + self.batch_size > T:
                    b_idx = idx[i:T]
                else:
                    b_idx = idx[i:i+self.batch_size]
                valid_b = valid[b_idx]
                
                new_log_probs, new_values, entropy = self.get_log_prob_value_entropy(observations[b_idx], actions[b_idx])
                logratio = (new_log_probs - log_probs[b_idx])
                ratio = logratio.exp()
                b_advantage = advantage[b_idx]
                b_returns = returns[b_idx]

                surrogate_loss_1 = ratio*b_advantage
                surrogate_loss_2 = torch.clamp(ratio,min= 1-self.eps,max = 1+self.eps)*b_advantage
                actor_loss_tensor = -torch.min(surrogate_loss_1, surrogate_loss_2)-self.entropy_coeff*entropy

                critic_loss_unclipped = (b_returns - new_values).pow(2)
                value_clipped = values[b_idx] + torch.clamp(new_values-values[b_idx], -self.eps, self.eps)
                critic_loss_clipped = (b_returns - value_clipped).pow(2)
                critic_loss_tensor = torch.max(critic_loss_clipped, critic_loss_unclipped)

                loss = (actor_loss_tensor[valid_b] + self.critic_coeff*critic_loss_tensor[valid_b]).mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()

                # we measure an approximation of the KL divergence between the updated and reference policy and we exit if it gets too large
                with torch.no_grad(): 
                    approx_kl_sum += ((ratio - 1) - logratio)[valid_b].mean().item()
                    kl_count += 1
                    approx_kl = approx_kl_sum / kl_count
                if kl_count>=(nb_batches//2) and approx_kl > self.target_KL : return



