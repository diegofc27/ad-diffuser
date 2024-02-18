import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import numpy as np
from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)
import math
from torch.distributions import Bernoulli


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()
        print("Attention: ", attention)
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            
            # print("x.shape, h[-1].shape")
            # print(x.shape, h[-1].shape)
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class ValueFunction(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )


    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
    
def create_discrete_matrices(Ts: float, num_moments: int = 4):
    full_row = np.array([1 / math.factorial(n) * Ts**n for n in range(num_moments)])
    dm = np.zeros((num_moments, num_moments))

    for idx in range(num_moments):
        dm[idx, idx:] = full_row[: num_moments - idx]

    db = np.array([1 / math.factorial(n) * Ts**n for n in range(num_moments, 0, -1)])

    return dm, db

class LongDynamics(nn.Module):
    """Longitudinal dynamics."""

    def __init__(self):
        """Initialize the LongDynamics module.

        Args:
            cfg (LongConfig): Configuration for longitudinal dynamics.
        """
        super().__init__()
        self.Ts = 0.2
        dm, db = create_discrete_matrices(self.Ts, 4)
        self.register_buffer("dm", torch.tensor(dm, dtype=torch.float32, device="cuda:0"))
        self.register_buffer("db", torch.tensor(db, dtype=torch.float32, device="cuda:0"))

    def forward(self, x: torch.Tensor, u: torch.Tensor):
        """Forward pass through the longitudinal dynamics module.

        Args:
            x (torch.Tensor): Input state tensor.
            u (torch.Tensor): Input control tensor.

        Returns:
            torch.Tensor: Output state tensor.
        """
        self.dm = self.dm.to(x.device)
        self.db = self.db.to(x.device)
        return torch.einsum("bj,ij->bi", x, self.dm) + self.db * u

    
class ValueFunctionL2(nn.Module):
    def __init__(
        self,
        horizon=1,
        transition_dim=16,
        dim=32,
        out_dim=1,
        cond_dim=2,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.dim = dim
        self.out_dim = out_dim
        self.l2 = nn.MSELoss()
        self.bmw_dynamics = LongDynamics()
    
    def forward(self, x, *args):
        '''
            x : [ batch x horizon x transition ]
            action : [ batch x horizon x action ]
        '''
        # import pdb; pdb.set_trace()
        x = einops.rearrange(x, 'b h t -> b t h')
        #add action to the transition dim
        # actions = x[:, :, :2]
        # theta = x[:, :, 2]
        # #add action to the transition dim
        # x_add = torch.zeros_like(x)
        # y_add = torch.zeros_like(x)
        # theta_add = torch.zeros_like(x)

        # x_,y_,theta_ = self.dynamics(actions[:,:,0], actions[:,:,1], theta)
        # x_add[:,:,0] = x_
        # y_add[:,:,1] = y_
        # theta_add[:,:,2] = theta_

        # x_1 = self.bmw_dynamics(x[:,1:5,:], x[:,0,:])
        #calculate the l2 loss for each batch
        diff_batch = []
        N = x.shape[-1]
        for i in range(N-1):
            x_1 = self.bmw_dynamics(x[:,1:5,i], x[:,0,i].unsqueeze(1))
            diff_batch.append(-self.l2(x[:,:,i+1][:,1:5], x_1[i]))
        # for i in range(len(x)-1):
        #     diff_batch.append(-self.l2(x[i+1], x_1[i]))
        #     import pdb; pdb.set_trace()
        #do this but in one operation
        diff_batch = torch.stack(diff_batch)
        # diff = self.l2(x[:,:,1:], x_1[:,:,:-1])
        return diff_batch
    
    
    def dynamics_safe_grid(self, force_left, force_right, theta, dt=1, wheelbase=2.5):
        # Normalize forces to ensure they are between 0 and 1
        max_force = torch.max(torch.abs(force_left), torch.abs(force_right))
        mask = max_force > 0
        force_left = torch.where(mask, force_left / max_force, force_left.clone())
        force_right = torch.where(mask, force_right / max_force, force_right.clone())


        # Calculate linear and angular velocities
        v_linear = 0.5 * (force_left + force_right)
        omega = (force_right - force_left) / wheelbase

        # Update state
        x = v_linear * torch.cos(theta) * dt
        y = v_linear * torch.sin(theta) * dt
        theta = (theta + omega * dt) % (2 * torch.pi)

        return x, y, theta



class MLPnet(nn.Module):
    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        horizon=1,
        returns_condition=True,
        skills_condition=False,
        condition_dropout=0.1,
        calc_energy=False,
        big_net=False,
        attention=False,
    ):
        super().__init__()

        if calc_energy:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.skill_condition = skills_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        self.transition_dim = transition_dim
        self.action_dim = transition_dim - cond_dim

        # if self.returns_condition:
        #     self.returns_mlp = nn.Sequential(
        #                 nn.Linear(1, dim),
        #                 act_fn,
        #                 nn.Linear(dim, dim * 4),
        #                 act_fn,
        #                 nn.Linear(dim * 4, dim),
        #             )
        #     self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
        #     embed_dim = 2*dim
        # elif self.skill_condition:
        #     self.skills_mlp = nn.Sequential(
        #                 nn.Linear(1, dim),
        #                 act_fn,
        #                 nn.Linear(dim, dim * 4),
        #                 act_fn,
        #                 nn.Linear(dim * 4, dim),
        #             )
        #     self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
        #     embed_dim = 2*dim
        # else:
        embed_dim = dim

         
        self.mlp = nn.Sequential(
                    nn.Linear(embed_dim + transition_dim, 1024),
                    act_fn,
                    nn.Linear(1024, 1024),
                    act_fn,
                    nn.Linear(1024, 1024),
                    act_fn,
                    nn.Linear(1024, 1024),
                    act_fn,
                    nn.Linear(1024, self.action_dim),
                )

    def forward(self, x, cond, time, returns=None, skills=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            returns : [batch x 1]
        '''
        # Assumes horizon = 1
        t = self.time_mlp(time)

        # if self.returns_condition:
        #     assert returns is not None
        #     returns_embed = self.returns_mlp(returns)
        #     if use_dropout:
        #         mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
        #         returns_embed = mask*returns_embed
        #     if force_dropout:
        #         returns_embed = 0*returns_embed
 
        #     t = torch.cat([t, returns_embed], dim=-1)

        
        # elif self.skill_condition:
        #     assert skills is not None
        #     skills_embed = self.skills_mlp(skills)
        #     if use_dropout:
        #         mask = self.mask_dist.sample(sample_shape=(skills_embed.size(0), 1)).to(skills_embed.device)
        #         skills_embed = mask*skills_embed
        #     if force_dropout:
        #         skills_embed = 0*skills_embed
        #     t = torch.cat([t, skills_embed], dim=-1)

        inp = torch.cat([t, cond, x], dim=-1)
        out  = self.mlp(inp)

        if self.calc_energy:
            energy = ((out - x) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x, create_graph=True)
            return grad[0]
        else:
            return out

class ValueFunctionH400(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            #nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Linear(544, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )


    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out


if __name__ == '__main__':
    #model = ValueFunctionL2()
    # x :[`batch_size` x `horizon` x `transition_dim`]
    x = torch.zeros(3, 1, 16)
    # action :[`batch_size` x `horizon` x `action_dim`]
    action = torch.randn(3, 1, 2)
    print(x)
    x[:, :, 2:] =+ action
    print(x)