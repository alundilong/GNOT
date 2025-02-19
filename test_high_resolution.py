#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import pickle
import torch
import numpy as np
import torch.nn as nn
import dgl
import matplotlib.pyplot as plt
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import get_seed, get_num_params
from args import get_args
from data_utils import get_test_dataset, get_model, get_loss_func, MIODataLoader
from train import validate_epoch
from utils import plot_heatmap, plot_time_sequence_heatmaps
import matplotlib.animation as animation
import os

if __name__ == "__main__":

    ckpt_dir_or_file = './hidden256/checkpoints/'
    with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
        ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    model_path = ckpt_path
    result = torch.load(model_path,map_location='cpu')

    args = result['args']

    model_dict = result['model']

    vis_component = 0 if args.component == 'all' else int(args.component)

    total_frames = 10
    res = 50
    npoints =res*res+4*res
    vis_component = 4

    device = torch.device('cpu')

    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    #test_path = "/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_long_time10_res50/porousmelting_train.pkl"
    #test_path = "/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_time10_res50/porousmelting_train.pkl"
    #test_path = "/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/pickle_time10_res100/porousmelting_train.pkl"
    test_path = "/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/regular_pickle_time10_res50/porousmelting_train.pkl"
    #test_path = "/home/maoy/data/PorousMedia/meltingFoam/DL_workspace/data/regular_pickle_time10_res100/porousmelting_train.pkl"
    test_dataset = get_test_dataset(args,test_path)

    test_sampler = SubsetRandomSampler(torch.arange(len(test_dataset)))

    test_loader = MIODataLoader(test_dataset, sampler=test_sampler, batch_size=1, drop_last=False)

    model = get_model(args,)

    model.load_state_dict(model_dict)

    model.eval()
    with torch.no_grad():
        #### test single case
        idx = 50
        g, u_p, g_u =  list(iter(test_loader))[idx]
        # u_p = u_p.unsqueeze(0)      ### test if necessary
        out = model(g, u_p, g_u)

        if args.x_normalizer is not None:
            g.ndata['x'] = args.x_normalizer.transform(g.ndata['x'],inverse=True)
        if args.y_normalizer is not None:
            g.ndata['y'] = args.y_normalizer.transform(g.ndata['y'],inverse=True)
            out = args.y_normalizer.transform(out,inverse=True)

        x0, y0 = g.ndata['x'][:,0].cpu().numpy(), g.ndata['x'][:,1].cpu().numpy()
        pred0 = out[:,vis_component].squeeze().cpu().numpy()
        target0 = g.ndata['y'][:,vis_component].squeeze().cpu().numpy()
        err0 = pred0 - target0
        print(f'size of xcoord: {len(x0)}')
        print(f'size of ycoord: {len(y0)}')
        print(f'size of pred: {len(pred0)}')
        print(f'size of target: {len(target0)}')
        print(f'size of err: {len(err0)}')
        print(np.linalg.norm(err0)/np.linalg.norm(target0))

        #### choose one to visualize
        cm = plt.get_cmap('rainbow')

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 12))

        sc_pred = axes[0].scatter([], [], cmap=cm,s=2)
        ax = axes[0]
        axes[0].set_title('pred')

        sc_target = axes[1].scatter([], [], s=2,cmap=cm)
        ax = axes[1]
        axes[1].set_title('target')

        sc_err = axes[2].scatter([], [], cmap=cm,s=2)
        ax = axes[2]
        axes[2].set_title('err')

        # Create colorbars
        cbar_pred = fig.colorbar(sc_pred, ax=axes[0])
        cbar_target = fig.colorbar(sc_target, ax=axes[1])
        cbar_err = fig.colorbar(sc_err, ax=axes[2])

        def update(frame):
            print(f'....................> Frame {frame}')  # Debugging output

            x = x0[frame * npoints:(frame + 1) * npoints]
            y = y0[frame * npoints:(frame + 1) * npoints]
            pred = pred0[frame * npoints:(frame + 1) * npoints]
            target = target0[frame * npoints:(frame + 1) * npoints]
            err = err0[frame * npoints:(frame + 1) * npoints]

            # Update scatter plots
            sc_pred.set_offsets(np.column_stack((x, y)))
            sc_pred.set_array(pred)

            sc_target.set_offsets(np.column_stack((x, y)))
            sc_target.set_array(target)

            sc_err.set_offsets(np.column_stack((x, y)))
            sc_err.set_array(err)

            # Dynamically update axes limits
            for ax in axes:
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())

            # Update colorbar limits dynamically
            cbar_pred.mappable.set_clim(pred.min(), pred.max())
            cbar_target.mappable.set_clim(target.min(), target.max())
            cbar_err.mappable.set_clim(err.min(), err.max())

            return sc_pred, sc_target, sc_err

        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)

        plt.show()
        print(f'....................> {vis_component}')

