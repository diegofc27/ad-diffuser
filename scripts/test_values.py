import diffuser.utils as utils
import torch
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)
#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#



## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()


diffusion = value_experiment.diffusion
dataset = value_experiment.dataset
renderer = value_experiment.renderer
model = value_experiment.model
trainer = value_experiment.trainer

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide

logger = logger_config()


#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#




#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, info = diffusion.loss(*batch)
print("loss",loss)
print("info",info)
print('✓')

# dataloader = utils.training.cycle(torch.utils.data.DataLoader(
#             dataset, batch_size=512, num_workers=4, shuffle=True, pin_memory=True
#         ))
# n_eval_steps =10
# with torch.no_grad():
#      for step in range(n_eval_steps):
#         batch = next(dataloader)
#         #import pdb; pdb.set_trace()

#         # print("Batch dim: ", batch.shape)
#         batch = batch_to_device(batch)

#         loss, infos = diffusion.loss(*batch)
#         print("loss",loss)
#         print("info ",info)       
#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#


trainer.test(n_test_steps=30)
