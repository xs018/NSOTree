from pycox import models
import torch
import torchtuples as tt
from torchtuples.tupletree import tuplefy
import numpy as np
# from torch.optim.lr_scheduler import CosineAnnealingLR
import torchtuples.callbacks as cb
from models.network_tree import NeuralTree
from timm.scheduler.cosine_lr import CosineLRScheduler

import numpy as np
from pycox import models
import torchtuples as tt
from torchtuples.utils import is_dl


class CoxPHNSTree(models.CoxPH):
    def __init__(self, 
                input_dim, 
                 output_dim, 
                 hidden_dim, 
                 num_layer, 
                 num_back_layer, 
                 backward_dim = 256,
                 drop_type = 'none',
                 p_drop = 0, 
                 optimizer=None, 
                 device=None, 
                 loss=None):
        self.p_drop = p_drop
        net = NeuralTree(input_dim, output_dim, hidden_dim, num_layer, num_back_layer, backward_dim, drop_type)
        super().__init__(net, optimizer, device, loss)

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True, num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224, **kwargs):
        self.training_data = tt.tuplefy(input, target)
        self.val_data = val_data
        if target is not None:
            input = (input, target)
        self.dataloader = self.make_dataloader(input, batch_size, shuffle, num_workers, **kwargs)
        self.val_dataloader = val_data
        if (is_dl(val_data) is False) and (val_data is not None):
            self.val_dataloader = self.make_dataloader(
                val_data, val_batch_size, shuffle=False, num_workers=num_workers, **kwargs
            )
        log = self.fit_dataloader(self.dataloader, epochs, callbacks, verbose, metrics, self.val_dataloader, False)
        return log

    def finetuing(self, optimizer, epochs=1, callbacks=None, verbose=True, metrics=None):
        self.optimizer = optimizer
        self.compute_logrank_test()
        self.net.traversal_tree()
        # self.net.fix_tree()
        log = self.fit_dataloader(self.dataloader, epochs, callbacks, verbose, metrics, self.val_dataloader, True)
        return log

    def fit_dataloader(
        self, dataloader, epochs=1, callbacks=None, verbose=True, metrics=None, val_dataloader=None, finetuing=False,
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

        stop = self.callbacks.on_fit_start()
        for _ in range(epochs):
            if stop:
                break
            stop = self.callbacks.on_epoch_start()
            if stop:
                break
            for data in dataloader:
                stop = self.callbacks.on_batch_start()
                if stop:
                    break
                self.optimizer.zero_grad()
                self.batch_metrics = self.compute_metrics(data, self.metrics, finetuing=finetuing)
                self.batch_loss = self.batch_metrics["loss"]
                self.batch_loss.backward()
                stop = self.callbacks.before_step()
                if stop:
                    break
                self.optimizer.step()
                stop = self.callbacks.on_batch_end()
                if stop:
                    break
            else:
                stop = self.callbacks.on_epoch_end()
        self.callbacks.on_fit_end()

        # # <--------------- do longrank_test --------------->
        # self.compute_logrank_test()
        # self.net.traversal_tree()
        # self.net.fix_tree()

        # # <--------------- start fine tuning --------------->
        # stop = self.callbacks.on_fit_start()
        # for _ in range(epochs):
        #     if stop:
        #         break
        #     stop = self.callbacks.on_epoch_start()
        #     if stop:
        #         break
        #     for data in dataloader:
        #         stop = self.callbacks.on_batch_start()
        #         if stop:
        #             break
        #         self.optimizer.zero_grad()
        #         self.batch_metrics = self.compute_metrics(data, self.metrics, finetuing=True)
        #         self.batch_loss = self.batch_metrics["loss"]
        #         self.batch_loss.backward()
        #         stop = self.callbacks.before_step()
        #         if stop:
        #             break
        #         self.optimizer.step()
        #         stop = self.callbacks.on_batch_end()
        #         if stop:
        #             break
        #     else:
        #         stop = self.callbacks.on_epoch_end()
        # self.callbacks.on_fit_end()

    def compute_logrank_test(self):
        input, target = self.val_data
        x = input
        t, e = target
        x = torch.from_numpy(x).to(self.device)
        _ = self.net(x, t, e, p=self.p_drop, finetuing=False)

    def compute_metrics(self, data, metrics=None, finetuing=False):
        if metrics is None:
            metrics = self.metrics
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to set `self.loss`.")

        input, target = data
        input = self._to_device(input)
        target = self._to_device(target)
        x = input[0]
        # t, e = target
        out = self.net(x, p=self.p_drop, finetuing=finetuing)
        
        out = tuplefy(out)
        return {name: metric(*out, *target) for name, metric in metrics.items()}


class CoxCCNSTree(models.CoxCC):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim, 
                 num_layer, 
                 num_back_layer, 
                 backward_dim = 256,
                 drop_type = 'none',
                 p_drop = -1,
                 optimizer=None, 
                 device=None, 
                 shrink=0, 
                 loss=None):
        self.p_drop = p_drop
        net = NeuralTree(input_dim, output_dim, hidden_dim, num_layer, num_back_layer, backward_dim, drop_type)
        super().__init__(net, optimizer, device, shrink, loss)

    def fit(self, input, target, batch_size=256, epochs=1, callbacks=None, verbose=True, num_workers=0, shuffle=True, metrics=None, val_data=None, val_batch_size=8224, n_control=1, shrink=None, **kwargs):
        if target is not None:
            input = (input, target)
        dataloader = self.make_dataloader(input, batch_size, shuffle, num_workers, **kwargs)
        val_dataloader = val_data
        if (is_dl(val_data) is False) and (val_data is not None):
            val_dataloader = self.make_dataloader(
                val_data, val_batch_size, shuffle=False, num_workers=num_workers, **kwargs
            )
        log = self.fit_dataloader(dataloader, epochs, callbacks, verbose, metrics, val_dataloader)
        return log

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

        stop = self.callbacks.on_fit_start()
        for _ in range(epochs):
            if stop:
                break
            stop = self.callbacks.on_epoch_start()
            if stop:
                break
            for data in dataloader:
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
            else:
                stop = self.callbacks.on_epoch_end()
        self.callbacks.on_fit_end()
        return self.log

    def compute_metrics(self, input, metrics):
        print(input)
        if (self.loss is None) and (self.loss in metrics.values()):
            raise RuntimeError(f"Need to specify a loss (self.loss). It's currently None")
        input = self._to_device(input)
        batch_size = input.lens().flatten().get_if_all_equal()
        if batch_size is None:
            raise RuntimeError("All elements in input does not have the same length.")
        case, control = input # both are TupleTree
        print(control)
        input_all = tt.TupleTree((case,) + control).cat()
        g_all = self.net(*input_all)
        g_all = tt.tuplefy(g_all).split(batch_size).flatten()
        g_case = g_all[0]
        g_control = g_all[1:]
        res = {name: metric(g_case, g_control) for name, metric in metrics.items()}
        return res