from torch.optim.optimizer import Optimizer, required
import torch

def NnueOptimizer(optimizer_cls, params, **kwargs):
    class SpecificNnueOptimizer(optimizer_cls):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            self.state['nnue_optimizer']['num_finished_steps'] = 0
            self.post_step_callback = None

            self._add_default('maskable', False)



        def set_post_step_callback(self, callback):
            self.post_step_callback = callback



        def get_num_finished_steps(self):
            return self.state['nnue_optimizer']['num_finished_steps']



        def freeze_parameter_region(self, param, indices):
            if len(indices) != len(param.shape):
                raise Exception('Invalid indices for parameter region.')

            if param not in self.state:
                raise Exception('No state for parameter.')

            state = self.state[param]

            if 'weight_mask' not in state:
                raise Exception('Parameter not masked.')

            state['weight_mask'][indices].fill_(0.0)



        def step(self, closure=None):
            loss = super(SpecificNnueOptimizer, self).step(closure)

            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]  # get state dict for this param

                    if not 'nnue_optimizer_initialized' in state:
                        state['nnue_optimizer_initialized'] = True

                        if group['maskable']:
                            state['weight_mask'] = torch.ones_like(p)

                    if 'weight_mask' in state:
                        p.data.mul_(state['weight_mask'])

            self.state['nnue_optimizer']['num_finished_steps'] += 1

            if self.post_step_callback is not None:
                self.post_step_callback(self)

            return loss



        def _add_default(self, default_name, default_value):
            if default_name in self.defaults:
                raise Exception('Default already exists.')

            self.defaults[default_name] = default_value

            for group in self.param_groups:
                if default_value is required and not default_name in group:
                    raise ValueError("parameter group didn't specify a value of required optimization parameter " + name)
                else:
                    group.setdefault(default_name, default_value)



    return SpecificNnueOptimizer(params, **kwargs)
