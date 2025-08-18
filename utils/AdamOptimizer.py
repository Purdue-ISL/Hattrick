import torch
from torch.optim.optimizer import Optimizer

class ADAMOptimizer(torch.optim.Optimizer):
    """
    implements ADAM Algorithm, as a preceding step.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ADAMOptimizer, self).__init__(params, defaults)
    def step(self):
        """
        Perform a single optimization step.
        """
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Momentum (Exponential MA of gradients)
                    state['exp_avg'] = torch.zeros_like(p.data, device=p.device)

                    # RMS Prop componenet. (Exponential MA of squared gradients). Denominator.
                    state['exp_avg_sq'] = torch.zeros_like(p.data, device=p.device)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                b1, b2 = group['betas']
                state['step'] += 1

                # Add weight decay if any
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, group['weight_decay'])

                # Momentum
                exp_avg = torch.mul(exp_avg, b1) + (1 - b1)*grad
                
                # RMS
                exp_avg_sq = torch.mul(exp_avg_sq, b2) + (1-b2)*(grad*grad)

                mhat = exp_avg / (1 - b1 ** state['step'])
                vhat = exp_avg_sq / (1 - b2 ** state['step'])
                
                denom = torch.sqrt(vhat) + group['eps']

                p.data = p.data - group['lr'] * mhat / denom 
                
                # Save state
                state['exp_avg'], state['exp_avg_sq'] = exp_avg, exp_avg_sq 

        return loss