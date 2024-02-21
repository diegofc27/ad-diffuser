import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,
        'log_freq': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 5000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:6',
        'seed': None,

        'run_name': 'diffusion',
        'run_group': 'diffusion',
    },

    'diffusion-ad': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'use_normalizer': True,
        'equal_weight': False,

        ## dataset
        'loader': 'datasets.ContextDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,
        'context_len': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'log_freq': 100,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 10000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:2',
        'seed': None,

        'run_name': 'diffusion-ad',
        'run_group': 'diffusion',
    },

    'diffusion-bmw': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 2,
        'loss_weights': None,
        'loss_discount': 1,
        'equal_action_weight':True,
        'predict_epsilon': True,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'equal_weight': False,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'use_normalizer': False,
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 32,
        'context_len': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/delta',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 100000,
        'batch_size': 256,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 20000,
        'log_freq': 100,
        'test_freq': 5000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'diffusion-bmw',
        'run_group': 'thesis',
    },

    'diffusion-action-bmw': {
        ## model
        'model': 'models.MLPnet',
        'diffusion': 'models.ActionGaussianDiffusion',
        'horizon': 1,
        'n_diffusion_steps': 20,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'equal_action_weight':True,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'equal_weight': False,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'use_normalizer': True,
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 32,
        'context_len': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/action',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 100000,
        'batch_size': 256,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 20000,
        'log_freq': 100,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'diffusion-action-bmw',
        'run_group': 'thesis',
    },

    'diffusion-action-stateLoss-bmw': {
        ## model
        'model': 'models.MLPnet',
        'diffusion': 'models.ActionGaussianDiffusion',
        'horizon': 10,
        'n_diffusion_steps': 20,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'equal_action_weight':True,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'equal_weight': False,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'use_normalizer': True,
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 32,
        'context_len': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/action',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 100000,
        'log_freq': 10,
        'batch_size': 256,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 20000,
        'test_freq': 100,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'diffusion-action-stateLoss-bmw',
        'run_group': 'thesis',
    },

    'diffusion-DynNet-bmw': {
        ## model
        'model': 'models.DynNet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'equal_action_weight':True,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'equal_weight': False,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'use_normalizer': True,
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 32,
        'context_len': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/action',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 100000,
        'log_freq': 10,
        'batch_size': 64,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 20000,
        'test_freq': 200,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'diffusion-dynet-bmw_10H',
        'run_group': 'thesis',
    },

     
    'diffusion-safe': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 16,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'use_normalizer': True,
        'equal_weight': False,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 50,
        'context_len': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 256,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'log_freq': 100,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'diffusion-safe',
        'run_group': 'thesis',
    },

     'diffusion-not-norm': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 16,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        'use_normalizer': False,
        'equal_weight': False,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 30,
        'context_len': 100,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'diffusion-not-norm',
        'run_group': 'thesis',
    },


    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 16,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8,16),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 30,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 512,
        'learning_rate': 1e-3, #2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'safe_grid_values',
        'run_group': 'values',
    },

    'values_cost': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 16,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8, 16),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.CostDataset',
        'normalizer': 'GaussianNormalizer',
        "safe_traj": True,
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 30,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/cost_nodiscount',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 512,
        'learning_rate': 1e-3, #2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'safe_grid_values',
        'run_group': 'values',
    },

    'values_safe': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 16,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8, 16),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.SafeDataset',
        'normalizer': 'GaussianNormalizer',
        "safe_traj": True,
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 30,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/safe_sigmoid',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 512,
        'learning_rate': 1e-3, #2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'safe_grid_values',
        'run_group': 'values',
    },


    'values_static_l2': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 1,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8,16),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': 0,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 30,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 512,
        'learning_rate': 1e-3, #2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'safe_grid_values',
        'run_group': 'values',
    },

    'values_static_l2_bmw': {
        'model': 'models.ValueFunctionL2',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 1,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8, 16),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': 0,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 32,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 512,
        'learning_rate': 1e-3, #2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,

        'run_name': 'bmw_values',
        'run_group': 'values',
    },

    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 50,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda:2',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 16,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/cost_nodiscount_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },

    'plan_l2': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda:2',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 16,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },

    'plan_l2_bmw': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 32,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda:2',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        #dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'use_normalizer': True,
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 32,
        'context_len': 100,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },

    'plan-ad': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.ContextPolicy',
        'max_episode_length': 500,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda:2',
        'seed': None,
        'guided':True,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
}


#------------------------ overrides ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}
