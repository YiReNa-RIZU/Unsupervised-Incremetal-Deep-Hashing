import torch
import torchvision
import torch.optim as optim
import os
import time
import models.alexnet as alexnet
import utils.evaluate as evaluate
import numpy as np

from loguru import logger
from models.adsh_loss import ADSH_Loss,ADUH_Loss
from data.data_loader import sample_dataloader
from draw_plot import loss_plot
from resnet import resnet20
from scipy.stats import norm


def train(
        query_dataloader,
        retrieval_dataloader,
        code_length,
        device,
        lr,
        max_iter,
        max_epoch,
        num_samples,
        batch_size,
        root,
        dataset,
        gamma,
        topk,
):
    """
    Training model.

    Args
        query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.
        gamma(float): Hyper-parameters.
        topk(int): Topk k map.

    Returns
        mAP(float): Mean Average Precision.
    """
    print('start dataset1 supervised step(1)')
    # Initialization
    model = alexnet.load_model(code_length).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    criterion = ADUH_Loss(code_length, gamma)
    hist_map = []
    hist_loss = []

    train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, num_samples, batch_size, root, dataset)
    print('query num:',len(query_dataloader.dataset))
    print('database num:',len(retrieval_dataloader.dataset))
    print('train num:',len(train_dataloader.dataset))
    
    num_retrieval = len(retrieval_dataloader.dataset)
    U = torch.zeros(num_samples, code_length).to(device)
    B = torch.randn(num_retrieval, code_length).to(device)
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    best_mAP = 0
    

    start = time.time()
    for it in range(max_iter):
        iter_start = time.time()
        
        
        
        # Sample training data for cnn learning
        with torch.no_grad():
            train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, num_samples, batch_size, root, dataset)
    
            # Create Similarity matrix
            train_targets = train_dataloader.dataset.get_onehot_targets().to(device)
            S = (train_targets @ retrieval_targets.t() > 0).float()
            S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1)).to(device)
    
            # Soft similarity matrix, benefit to converge
            r = S.sum() / (1 - S).sum()
            S = S * (1 + r) - r
        
        # Training CNN model
        for epoch in range(max_epoch):
            for batch, (data, targets, index) in enumerate(train_dataloader):
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                optimizer.zero_grad()

                F = model(data)
                U[index, :] = F.data
                cnn_loss = criterion(F, B, S[index, :], sample_index[index])

                cnn_loss.backward()
                optimizer.step()

        # Update B
        expand_U = torch.zeros(B.shape).to(device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, gamma)

        # Total loss
        iter_loss = calc_loss(U, B, S, code_length, sample_index, gamma)
        hist_loss.append(iter_loss)
        print('[iter:{}/{}][loss:{:.2f}][iter_time:{:.2f}]'.format(it+1, max_iter, iter_loss, time.time()-iter_start))
        logger.debug('[iter:{}/{}][loss:{:.2f}][iter_time:{:.2f}]'.format(it+1, max_iter, iter_loss, time.time()-iter_start))
        
        query_code = generate_code(model, query_dataloader, code_length, device)
        mAP = evaluate.mean_average_precision(
            query_code.to(device),
            B,
            query_dataloader.dataset.get_onehot_targets().to(device),
            retrieval_targets,
            device,
            topk,
        )
        model.train()
        logger.info('map:{:.4f}'.format(mAP))
        hist_map.append(mAP)
        
        if mAP > best_mAP:
            best_mAP = mAP
            model.eval()
            print('save data for best map is %.4f of epoch %d'%(best_mAP,it+1))
            # Save checkpoints
            torch.save(query_code.cpu(), os.path.join('checkpoints', 'query_code1_down.t'))
            torch.save(B.cpu(), os.path.join('checkpoints', 'B_dataset1_down.t'))
            torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join('checkpoints', 'query_targets.t'))
            torch.save(retrieval_targets.cpu(), os.path.join('checkpoints', 'database_targets.t'))
            torch.save(model.cpu(), os.path.join('checkpoints', 'model_step1_down.t'))
            torch.save(best_mAP, os.path.join('checkpoints', 'best_map_down_data1.t'))
            model.cuda()
            model.train()
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))

    # Evaluate
    query_code = generate_code(model, query_dataloader, code_length, device)
    mAP = evaluate.mean_average_precision(
        query_code.to(device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(device),
        retrieval_targets,
        device,
        topk,
    )
    model.train()


    loss_plot(hist_loss,'./','data1_down_1')
    
    
    
    
    
    model = torch.load(os.path.join('checkpoints', 'model_step1_down.t'))
    best_mAP = torch.load(os.path.join('checkpoints', 'best_map_down_data1.t')) 
    model.cuda()
    
    print('load model best_map is %.4f'%best_mAP)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
    torch.save(retrieval_code.cpu(), os.path.join('checkpoints', 'U_database1_down.t'))
    model.train()
    print('finish construct dataset1 down_code,size is %d x %d'%(retrieval_code.size(0), retrieval_code.size(1)))
    
    

    return mAP, hist_map


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    return code

