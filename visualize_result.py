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
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from train import validate_epoch
from utils import plot_heatmap, plot_time_sequence_heatmaps
import os

if __name__ == "__main__":

    ckpt_dir_or_file = './data/checkpoints/'
    with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
        ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    model_path = ckpt_path
    result = torch.load(model_path,map_location='cpu')

    args = result['args']

    model_dict = result['model']

    vis_component = 0 if args.component == 'all' else int(args.component)
    frame = 10
    npoints = 2500+200
    vis_component = 3

    device = torch.device('cpu')

    kwargs = {'pin_memory': False} if args.gpu else {}
    get_seed(args.seed, printout=False)

    train_dataset, test_dataset = get_dataset(args)

    test_sampler = SubsetRandomSampler(torch.arange(len(test_dataset)))

    test_loader = MIODataLoader(test_dataset, sampler=test_sampler, batch_size=1, drop_last=False)

    loss_func = get_loss_func(args.loss_name, args, regularizer=True,  normalizer=args.y_normalizer)
    metric_func = get_loss_func(args.loss_name, args , regularizer=False, normalizer=args.y_normalizer)

    model = get_model(args,)

    model.load_state_dict(model_dict)

    model.eval()
    with torch.no_grad():
        #### test single case
        idx = 0
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

        first_frame = 0
        second_frame = 5
        third_frame = 10
        frame = first_frame
        x = x0[frame*npoints:(frame+1)*npoints]
        y = y0[frame*npoints:(frame+1)*npoints]
        pred = pred0[frame*npoints:(frame+1)*npoints]
        target = target0[frame*npoints:(frame+1)*npoints]
        err = err0[frame*npoints:(frame+1)*npoints]

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 12))

        sc = axes[0,0].scatter(x, y, c=pred, cmap=cm,s=2)
        ax = axes[0,0]
        fig.colorbar(sc, ax=ax)
        axes[0,0].set_title('pred')

        sc = axes[1,0].scatter(x, y, c=target, s=2,cmap=cm)
        ax = axes[1,0]
        fig.colorbar(sc, ax=ax)
        axes[1,0].set_title('target')

        sc = axes[2,0].scatter(x, y, c=err, cmap=cm,s=2)
        ax = axes[2,0]
        fig.colorbar(sc, ax=ax)
        axes[2,0].set_title('err')

        frame = second_frame
        x = x0[frame*npoints:(frame+1)*npoints]
        y = y0[frame*npoints:(frame+1)*npoints]
        pred = pred0[frame*npoints:(frame+1)*npoints]
        target = target0[frame*npoints:(frame+1)*npoints]
        err = err0[frame*npoints:(frame+1)*npoints]

        sc = axes[0,1].scatter(x, y, c=pred, cmap=cm,s=2)
        ax = axes[0,1]
        fig.colorbar(sc, ax=ax)
        axes[0,1].set_title('pred')

        sc = axes[1,1].scatter(x, y, c=target, s=2,cmap=cm)
        ax = axes[1,1]
        fig.colorbar(sc, ax=ax)
        axes[1,1].set_title('target')

        sc = axes[2,1].scatter(x, y, c=err, cmap=cm,s=2)
        ax = axes[2,1]
        fig.colorbar(sc, ax=ax)
        axes[2,1].set_title('err')

        frame = third_frame
        x = x0[frame*npoints:(frame+1)*npoints]
        y = y0[frame*npoints:(frame+1)*npoints]
        pred = pred0[frame*npoints:(frame+1)*npoints]
        target = target0[frame*npoints:(frame+1)*npoints]
        err = err0[frame*npoints:(frame+1)*npoints]

        sc = axes[0,2].scatter(x, y, c=pred, cmap=cm,s=2)
        ax = axes[0,2]
        fig.colorbar(sc, ax=ax)
        axes[0,2].set_title('pred')

        sc = axes[1,2].scatter(x, y, c=target, s=2,cmap=cm)
        ax = axes[1,2]
        fig.colorbar(sc, ax=ax)
        axes[1,2].set_title('target')

        sc = axes[2,2].scatter(x, y, c=err, cmap=cm,s=2)
        ax = axes[2,2]
        fig.colorbar(sc, ax=ax)
        axes[2,2].set_title('err')

        plt.show()
        print(f'....................> {vis_component}')
    
        #exit(1)

        myIndex = 0
        mypred = out[:,myIndex].squeeze().cpu().numpy()
        mytarget = g.ndata['y'][:,myIndex].squeeze().cpu().numpy()
        plot_time_sequence_heatmaps(x0,y0,mypred,mytarget,11,2700,title="Ux",path="Ux.png")

        myIndex = 1
        mypred = out[:,myIndex].squeeze().cpu().numpy()
        mytarget = g.ndata['y'][:,myIndex].squeeze().cpu().numpy()
        plot_time_sequence_heatmaps(x0,y0,mypred,mytarget,11,2700,title="Uy",path="Uy.png")

        myIndex = 2
        mypred = out[:,myIndex].squeeze().cpu().numpy()
        mytarget = g.ndata['y'][:,myIndex].squeeze().cpu().numpy()
        plot_time_sequence_heatmaps(x0,y0,mypred,mytarget,11,2700,title="prgh",path="prgh.png")

        myIndex = 3
        mypred = out[:,myIndex].squeeze().cpu().numpy()
        mytarget = g.ndata['y'][:,myIndex].squeeze().cpu().numpy()
        plot_time_sequence_heatmaps(x0,y0,mypred,mytarget,11,2700,title="T",path="T.png")

        myIndex = 4
        mypred = out[:,myIndex].squeeze().cpu().numpy()
        mytarget = g.ndata['y'][:,myIndex].squeeze().cpu().numpy()
        plot_time_sequence_heatmaps(x0,y0,mypred,mytarget,11,2700,title="alpha",path="alpha.png")

        plot_heatmap(x, y, pred,path='./pred.png',cmap=cm,show=True,title='pred')
        plot_heatmap(x, y, target,path='./target.png',cmap=cm,show=True,title='target')
