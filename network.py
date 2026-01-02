import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import LinearSVDO


# Define a simple 2 layer Network
class Net(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dims, alpha_threshold, theta_threshold, device):
        super(Net, self).__init__()
        self.fc1 = LinearSVDO(state_dim, hidden_dims[0], alpha_threshold, theta_threshold, device)
        self.fc2 = LinearSVDO(hidden_dims[0], hidden_dims[1], alpha_threshold, theta_threshold, device)
        self.fc3 = LinearSVDO(hidden_dims[1], action_dim, alpha_threshold, theta_threshold, device)
        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)
        self.device = device
        self.alpha_threshold = alpha_threshold

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, x):
        x = self._format(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = x * self.action_rescale + self.action_rescale_bias
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, student_hidden_dims, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, student_hidden_dims[0])
        self.l2 = nn.Linear(student_hidden_dims[0], student_hidden_dims[1])
        self.l3 = nn.Linear(student_hidden_dims[1], action_dim)
        self.device = 'cuda'

        self.max_action = max_action

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        a = F.relu(self.l1(x))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.device = 'cuda'

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        sa = torch.cat([x, u], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1