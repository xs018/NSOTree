import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import numpy as np
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
from sksurv.datasets import get_x_y

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from models.cox_nstree import CoxNOSTree

from utils.dataloading import load_dataset, format_dataset_to_df


class Concordance(tt.cb.MonitorMetrics):
    def __init__(self, x, durations, events, per_epoch=1, verbose=True):
        super().__init__(per_epoch)
        self.x = x
        self.durations = durations
        self.events = events
        self.verbose = verbose
    
    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            _ = self.model.compute_baseline_hazards()
            surv = self.model.predict_surv_df(self.x)
            ev = EvalSurv(surv, self.durations, self.events)
            concordance = ev.concordance_td()
            self.append_score('concordance', concordance)
            if self.verbose:
                print('c-index:', f"{concordance:.4f}")


class GatherMetrics(tt.cb.MonitorMetrics):
    def __init__(self, x, durations, events, per_epoch=1, verbose=True):
        super().__init__(per_epoch)
        self.x = torch.from_numpy(x).cuda()
        self.durations = durations
        self.events = events
        self.verbose = verbose
    
    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            # <------------ c-index ------------>
            _ = self.model.compute_baseline_hazards()
            surv = self.model.predict_surv_df(self.x)
            ev = EvalSurv(surv, self.durations, self.events)
            concordance = ev.concordance_td()
            self.append_score('c_index', concordance)

            # <------------ jacobian rank ------------>
            with torch.no_grad():
                jacobians, offsets = self.model.net(self.x, return_deriv=True)
            jacobians = jacobians.view(jacobians.shape[0], -1)
            jacobians, offsets = jacobians.cpu().numpy(), offsets.cpu().numpy()
            leaf_input = np.concatenate([jacobians, offsets], axis=1)

            jacobians_rank = matrix_rank(jacobians)
            offsets_rank = matrix_rank(offsets)
            leaf_input_rank = matrix_rank(leaf_input)
            
            self.append_score('jacobians_rank', jacobians_rank)
            self.append_score('offsets_rank', offsets_rank)
            self.append_score('leaf_input_rank', leaf_input_rank)

            if self.verbose:
                print('c-index:', f"{concordance:.4f}")
                print(f"c-index:{concordance:.4f}, jacobians rank:[{jacobians_rank}/{jacobians.shape[1]}] offsets_rank:[{offsets_rank}/{offsets.shape[1]}] leaf_input_rank:[{leaf_input_rank}/{leaf_input.shape[1]}]")

    def on_fit_end(self):
        print(np.max(self.scores['c_index']['score']))
        # print(self.model.net.soft_thres.lam)
        save_path = f"tree_plots/depth{self.model.net.num_layer}.jpg"
        # self.model.net.build_tree(self.x, self.durations, self.events, topk=2, save_path=save_path)
        # self.model.net.prune_tree(save_path=f"tree_plots/depth{self.model.net.num_layer}_pruned.jpg")


if __name__ == "__main__":
    dataset = 'simulated_linear'
    data = load_dataset(dataset)

    random_state = 3
    np.random.seed(random_state)
    _ = torch.manual_seed(random_state)

    vis = False
    hidden_dim = 1
    depth = 5
    back_n =  0
    drop_type = 'node_dropconnect'
    p_dropout = 0.1
    net_type = 'locally_constant'
    anneal = 'approx'
    batch_size = 128
    epochs = 100
    lr = 1e-3
    momentum = 0.95

    train_data, valid_data, test_data, viz_data, viz_data_unnorm = data['train'], data['valid'], data['test'], data['viz'], data['viz_unnorm']
    
    # train_data, valid_data = data['train'], data['val']
    train_data_df = format_dataset_to_df(train_data, 'T', 'E')
    X_train, y_train = get_x_y(train_data_df, attr_labels=['E', 'T'], pos_label=1)
    val_data_df = format_dataset_to_df(valid_data, 'T', 'E')
    X_valid, y_valid = get_x_y(val_data_df, attr_labels=['E', 'T'], pos_label=1)
    test_data_df = format_dataset_to_df(test_data, 'T', 'E')
    X_test, y_test = get_x_y(test_data_df, attr_labels=['E', 'T'], pos_label=1)

    get_target = lambda df: (df['T'].values, df['E'].values)
    y_train = get_target(train_data_df)
    y_valid = get_target(val_data_df)
    y_test = get_target(test_data_df)

    X_train = X_train.values.astype(np.float32)
    X_valid = X_valid.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32) # 1000x10

    # print(X_test.shape)

    durations_test, events_test = get_target(test_data_df)
    val = (X_valid, y_valid)

    input_dim = X_train.shape[1]
    output_dim = 1
    
    optimizer = tt.optim.SGD(lr=lr, momentum=momentum, weight_decay=0, nesterov=False)
    # optimizer = tt.optim.AdamW(lr=lr, decoupled_weight_decay=0)

    model = CoxNOSTree(input_dim=input_dim, 
                                output_dim=output_dim, 
                                hidden_dim=hidden_dim, 
                                num_layer=depth, 
                                num_back_layer=back_n, 
                                dense=True, 
                                drop_type=drop_type, 
                                net_type=net_type, 
                                approx=anneal,
                                p_drop = p_dropout,
                                optimizer=optimizer,
                                lr=lr)
    
    callbacks = [
                # tt.callbacks.EarlyStopping(), 
                #  tt.callbacks.ClipGradNorm(model.net, 12.0), 
                GatherMetrics(X_test, durations_test, events_test), 
                ]
    verbose = True

    log = model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)
    
    # _ = model.compute_baseline_hazards()
    # surv = model.predict_surv_df(X_test)
    # ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    # c_index = ev.concordance_td()
    # print(f"C-Index = {c_index:.4f}")

    # num_bootstrap = 100
    # _ = model.compute_baseline_hazards()
    # surv = model.predict_surv_df(X_test)
    # ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
    # c_index = ev.concordance_td()
    # print(f"C-Index = {c_index:.4f}")

    # C_indinces = []
    # indices = np.arange(len(X_test))
    # for _ in range(num_bootstrap):
    #         bootstraped_indices = np.random.choice(indices, size=len(X_test), replace=True)
    #         bootstrapped_X_test = X_test[bootstraped_indices]
    #         bootstrapped_y_test_T = durations_test[bootstraped_indices]
    #         bootstrapped_y_test_E = events_test[bootstraped_indices]
            
    #         surv = model.predict_surv_df(bootstrapped_X_test)
    #         ev = EvalSurv(surv, bootstrapped_y_test_T, bootstrapped_y_test_E, censor_surv='km')
    #         C_indinces.append(ev.concordance_td())
    
    # C_indinces = np.array(C_indinces)
    # c_index_lower = np.percentile(C_indinces, 2.5)
    # c_index_upper = np.percentile(C_indinces, 97.5)

    # print(f"C-Index (95% CI) = {c_index:.4f} ({c_index_lower:.3f}, {c_index_upper:.3f})")