import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants (customize for your case)
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
support_size = 10  # Used for reward/value categorical outputs — change if needed

class Representation(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.skip = torch.nn.Linear(input_dim, output_dim)
        self.layer1 = torch.nn.Linear(input_dim, width)
        self.layer2 = torch.nn.Linear(width, width)
        self.layer3 = torch.nn.Linear(width, width)
        self.layer4 = torch.nn.Linear(width, width)
        self.layer5 = torch.nn.Linear(width, output_dim)

    def forward(self, x):
        s = self.skip(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(x + s)
        x = 2 * (x - x.min(-1, keepdim=True)[0]) / (x.max(-1, keepdim=True)[0] - x.min(-1, keepdim=True)[0] + 1e-8) - 1
        return x

class Dynamics(nn.Module):
    def __init__(self, input_dim, output_dim, width, action_space):
        super().__init__()
        self.layer1 = nn.Linear(input_dim + action_space, width)
        self.layer2 = nn.Linear(width, width)
        self.hs_head = nn.Linear(width, output_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, support_size * 2 + 1)
        )
        self.one_hot_act = torch.cat([
            F.one_hot(torch.arange(0, action_space) % action_space, num_classes=action_space),
            torch.zeros(1, action_space)
        ], dim=0).to(device)

    def forward(self, x, action):
        action = self.one_hot_act[action.squeeze(1)]
        x = torch.cat((x, action.to(device)), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        hs = F.relu(self.hs_head(x))
        reward = self.reward_head(x)
        hs = 2 * (hs - hs.min(-1, keepdim=True)[0]) / (hs.max(-1, keepdim=True)[0] - hs.min(-1, keepdim=True)[0] + 1e-8) - 1
        return hs, reward

class Prediction(nn.Module):
    def __init__(self, input_dim, output_dim, width):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, width)
        self.layer2 = nn.Linear(width, width)
        self.policy_head = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, output_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, support_size * 2 + 1)
        )

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        P = F.softmax(self.policy_head(x), dim=-1)
        V = self.value_head(x)
        return P, V
