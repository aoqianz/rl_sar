import torch
import importlib
import torch.nn as nn


class Actor(nn.Module):
    def __init__(
        self,
        num_actor_obs,
        num_actions,
        actor_hidden_dims=[256, 256],
        activation=nn.ReLU(),
    ):
        super(Actor, self).__init__()
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, input):
        return self.actor(input)


model_class = Actor(num_actor_obs=55, num_actions=12)
ckpt = torch.load("/home/wang/Zihao/rl_sar/src/rl_sar/models/go2_xzh/model_1500.pt")[
    "model_state_dict"
]
load_rst = model_class.load_state_dict(ckpt["actor.0.weight"])
print(load_rst)
