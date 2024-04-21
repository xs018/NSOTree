from pycox import models
from torchtuples.tupletree import tuplefy
from torch.optim.lr_scheduler import StepLR
import torchtuples.callbacks as cb
from models.network import NeuralTree

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
        
        self.p_drop = p_drop
        self.alpha = alpha
        self.anneal = approx

        if loss is None:
            loss = models.loss.CoxPHLoss()
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

        self.lr_scheduler = StepLR(self.optimizer.optimizer, step_size=10, gamma=0.2)

        stop = self.callbacks.on_fit_start()
        for epoch in range(epochs):
            self.lr_scheduler.step()
            self.alpha = get_alpha(epoch, epochs)

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

    def compute_metrics(self, data, metrics=None):
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

        input, target = data
        input = self._to_device(input)
        target = self._to_device(target)
        
        if self.net.net_type == 'locally_constant':
            out = self.net(input[0], None, None, self.p_drop, self.alpha)
        elif self.net.net_type == 'locally_linear':
            out = self.net.normal_forward(*input)

        out = tuplefy(out)
        return {name: metric(*out, *target) for name, metric in metrics.items()}