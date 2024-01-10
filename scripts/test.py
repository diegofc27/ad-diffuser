import torch

x = torch.zeros(3, 1, 16)
# action :[`batch_size` x `horizon` x `action_dim`]
action = torch.randn(3, 1, 2)
print(x)
import pdb; pdb.set_trace()
x[:, :, 2:] =+ action
print(x)