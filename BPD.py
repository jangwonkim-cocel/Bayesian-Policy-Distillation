import copy
import torch
import torch.nn.functional as F
from SGVLB import SGVLB
from network import Net, Critic


class BPDAgent(object):
    def __init__(
            self,
            env,
            args,
            env_info,
            thresholds,
            datasize,
            device,
            discount,
            tau,
            noise_clip,
            policy_freq,
            h,
            num_teacher_param,
    ):
        self.args = args
        self.env = env
        self.env_info = env_info

        self.actor = Net(env_info['state_dim'], env_info['action_dim'], env_info['action_bound'],
                         args.student_hidden_dims, thresholds['ALPHA_THRESHOLD'], thresholds['THETA_THRESHOLD'],
                         device=device).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.sgvlb = SGVLB(self.actor, datasize, loss_type='l2', device=device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(env_info['state_dim'], env_info['action_dim']).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.datasize = datasize
        self.h = h

        self.total_it = 0
        self.kl_weight = 0

    def set_kl_weight(self, kl_weight):
        self.kl_weight = kl_weight
        return

    def test(self):
        self.actor.eval()
        with torch.no_grad():
            return_list = []
            for epi_cnt in range(1, self.args.num_test_epi):
                episode_return = 0
                done = False
                state, _ = self.env.reset()
                while not done:
                    action = self.actor(state)
                    action = action.cpu().numpy()[0]
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode_return += reward
                    state = next_state
                return_list.append(episode_return)

        avg_return = sum(return_list) / len(return_list)
        max_return = max(return_list)
        min_return = min(return_list)

        return avg_return, max_return, min_return

    def train(self, transition):
        self.actor.train()

        self.total_it += 1

        states, actions, rewards, next_states, dones = transition

        with torch.no_grad():
            next_actions = (
                    self.actor_target(next_states)
            ).clamp(self.env_info['action_bound'][0], self.env_info['action_bound'][1])

            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            Q = self.critic.Q1(states, pi)
            lmbda = (self.h * self.datasize) / Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + self.sgvlb(pi, actions, self.kl_weight)  # lambda = h*|D|/avg(|Q|)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def __del__(self):
        del self.actor
        del self.actor_target
        del self.critic
        del self.critic_target
        return

