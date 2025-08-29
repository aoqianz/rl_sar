import torch

# ckpt = torch.load("/home/wang/Zihao/rl_sar/src/rl_sar/models/go2_xzh/model_1500.pt")
# ckpt = torch.load("/home/wang/Zihao/rl_sar/src/rl_sar/models/go2_xzh/policy_1.pt")

# ckpt = torch.load("/home/wang/Zihao/rl_sar/src/rl_sar/models/go2_isaacgym/himloco.pt")

model = torch.jit.load("/home/wang/Zihao/rl_sar/src/rl_sar/models/go2_xzh/policy_1.pt")
print(model.code)

input_info = [inp.type() for inp in model.graph.inputs()]
print(input_info)