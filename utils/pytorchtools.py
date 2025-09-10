import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_model = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, current_loss, best_score, pretrain=True):
        if pretrain:
            if current_loss[0].avg < best_score['total_loss'] and current_loss[1] > best_score['acc']:
                self.save_model = True
                self.counter = 0
            else:
                self.save_model = False
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            # if current_loss[4].avg >= best_score['mask IoU'] and current_loss[3].avg >= best_score['boundary IoU'] and \
            #         current_loss[1].avg <= best_score['point_loss']+0.0005 and current_loss[2].avg <= best_score['match_loss']+0.01:
            if current_loss[4].avg >= best_score['mask IoU'] and current_loss[3].avg >= best_score['boundary IoU']:
                self.save_model = True
                self.counter = 0
            else:
                self.save_model = False
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
