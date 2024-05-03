from pycox import models
import torchtuples as tt
from torchtuples.tupletree import tuplefy
import numpy as np
# from torch.optim.lr_scheduler import CosineAnnealingLR
import torchtuples.callbacks as cb
from models.network import NeuralTree
from timm.scheduler.cosine_lr import CosineLRScheduler

def get_alpha(epoch, total_epoch):
    return float(epoch) / float(total_epoch)


class CoxNOSTree(models.cox._CoxPHBase):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim, 
                 num_layer, 
                 num_back_layer, 
                 dense = True, 
                 drop_type = 'none', 
                 p_drop = -1,
                 net_type = 'locally_constant', 
                 approx = 'none',
                 alpha = 1,
                 optimizer=None, 
                 lr=1e-4,
                 device=None, 
                 loss=None):
        
        net = NeuralTree(input_dim, 
                output_dim, 
                hidden_dim, 
                num_layer, 
                num_back_layer, 
                dense, 
                drop_type, 
                net_type, 
                approx)
        self.lr = lr
        self.p_drop = p_drop
        self.alpha = alpha
        self.anneal = approx

        if loss is None:
            loss = models.loss.CoxCCLoss(0.1)
        super().__init__(net, loss, optimizer, device)

    def fit_dataloader(
        self, dataloader, epochs=1, callbacks=None, verbose=True, metrics=None, val_dataloader=None
    ):
        self._setup_train_info(dataloader)
        self.metrics = self._setup_metrics(metrics)
        self.log.verbose = verbose
        self.val_metrics.dataloader = val_dataloader
        if callbacks is None:
            callbacks = []
        self.callbacks = cb.TrainingCallbackHandler(
            self.optimizer, self.train_metrics, self.log, self.val_metrics, callbacks
        )
        self.callbacks.give_model(self)

        # self.lr_scheduler = StepLR(self.optimizer.optimizer, step_size=10, gamma=0.2)
        
        num_steps = len(dataloader)
        warmup_steps = 10 * num_steps
        self.lr_scheduler = CosineLRScheduler(self.optimizer.optimizer, 
                                              t_initial=epochs*num_steps, 
                                              warmup_lr_init=self.lr,
                                              warmup_t=warmup_steps,
                                              cycle_limit=1,
                                              t_in_epochs=False,
                                              )

        stop = self.callbacks.on_fit_start()
        for epoch in range(epochs):
            self.alpha = get_alpha(epoch, epochs)

            if stop:
                break
            stop = self.callbacks.on_epoch_start()
            if stop:
                break
            for idx, data in enumerate(dataloader):
                stop = self.callbacks.on_batch_start()
                if stop:
                    break
                self.optimizer.zero_grad()
                self.batch_metrics = self.compute_metrics(data, self.metrics)
                self.batch_loss = self.batch_metrics["loss"]
                self.batch_loss.backward()
                stop = self.callbacks.before_step()
                if stop:
                    break
                self.optimizer.step()
                stop = self.callbacks.on_batch_end()
                if stop:
                    break
                self.lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                stop = self.callbacks.on_epoch_end()

        self.callbacks.on_fit_end()
        return self.log

    def compute_metrics(self, input, metrics=None):
        input, target = self._sorted_input_target(input, target)
        """Function for computing the loss and other metrics.

        Arguments:
            data {tensor or tuple} -- A batch of data. Typically the tuple `(input, target)`.

        Keyword Arguments:
            metrics {dict} -- A dictionary with metrics. If `None` use `self.metrics`. (default: {None})
        """
        if metrics is None:
            metrics = self.metrics
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to set `self.loss`.")

        # input, target = data
        # input = self._to_device(input)
        # target = self._to_device(target)
        
        # if self.net.net_type == 'locally_constant':
        #     out = self.net(input[0], None, None, self.p_drop, self.alpha)
        # elif self.net.net_type == 'locally_linear':
        #     out = self.net.normal_forward(*input)

        # out = tuplefy(out)
        # return {name: metric(*out, *target) for name, metric in metrics.items()}
        input = self._to_device(input)
        batch_size = input.lens().flatten().get_if_all_equal()
        if batch_size is None:
            raise RuntimeError("All elements in input does not have the same length.")
        case, control = input # both are TupleTree
        input_all = tt.TupleTree((case,) + control).cat()
        print(input_all)
        
        if self.net.net_type == 'locally_constant':
            g_all = self.net(input_all[0], None, None, self.p_drop, self.alpha)
        elif self.net.net_type == 'locally_linear':
            g_all = self.net.normal_forward(*input)

        g_all = tt.tuplefy(g_all).split(batch_size).flatten()
        g_case = g_all[0]
        g_control = g_all[1:]
        res = {name: metric(g_case, g_control) for name, metric in metrics.items()}
        return res
    
    @staticmethod
    def _sorted_input_target(input, target):
        input, target = tt.tuplefy(input, target).to_numpy()
        durations, _ = target
        idx_sort = np.argsort(durations)
        if (idx_sort == np.arange(0, len(idx_sort))).all():
            return input, target
        input = tt.tuplefy(input).iloc[idx_sort]
        target = tt.tuplefy(target).iloc[idx_sort]
        return input, target