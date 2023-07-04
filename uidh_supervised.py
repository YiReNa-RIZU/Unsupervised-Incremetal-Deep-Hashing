import torch
import torchvision
import torch.optim as optim
import os
import time
import models.alexnet as alexnet
import utils.evaluate as evaluate
import numpy as np

from loguru import logger
from models.adsh_loss import ADSH_Loss,ADUH_Loss,UIDH_Loss
from data.data_loader import sample_dataloader
from draw_plot import loss_plot
from resnet import resnet20
from scipy.stats import norm


def train(
        query_dataloader,
        retrieval_dataloader,
        query_dataloader_old,
        retrieval_dataloader_old,
        code_length,
        device,
        lr,
        max_iter,
        max_epoch,
        num_samples,
        batch_size,
        root,
        root2,
        dataset,
        dataset2,
        gamma,
        topk,
        topk2,
        loss_type,
        mu,
        ita,
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
    print('start dataset2 supervised step')
    print('down mode is: supervised')
    print('loss type is:%s'%loss_type)
    # Initialization
    #model = alexnet.load_model(code_length).to(device)
    model = torch.load(os.path.join('checkpoints', 'model_step1_down.t')).to(device)
    model.train()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )
    criterion = UIDH_Loss(code_length, gamma, mu, ita, loss_type)
    hist_map = []
    hist_loss = []

    train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, num_samples, batch_size, root2, dataset2)
    print('query num:',len(query_dataloader.dataset))
    print('database num:',len(retrieval_dataloader.dataset))
    print('train num:',len(train_dataloader.dataset))
    
    num_retrieval = len(retrieval_dataloader.dataset)
    U = torch.zeros(num_samples, code_length).to(device)
    B = torch.randn(num_retrieval, code_length).to(device)
    U_old_save = torch.zeros(num_samples,code_length).to(device)
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    best_mAP = 0
    
    
    #read dataset1 down code
    U_old = torch.load(os.path.join('checkpoints', 'U_database1_down.t'))
    U_old = U_old.detach().cuda()
    print(U_old.size())
    
    
    start = time.time()
    for it in range(max_iter):
        iter_start = time.time()
        
        
        #read sample data and similarity matrix
        with torch.no_grad():
            train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, num_samples, batch_size, root2, dataset2)
    
            # Create Similarity matrix
            train_targets = train_dataloader.dataset.get_onehot_targets().to(device)
            S = (train_targets @ retrieval_targets.t() > 0).float()
            S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1)).to(device)
    
            # Soft similarity matrix, benefit to converge
            r = S.sum() / (1 - S).sum()
            S = S * (1 + r) - r      
        train_dataloader_old, sample_index_old = sample_dataloader(retrieval_dataloader_old, num_samples, batch_size, root, dataset)
        
        
        # Training CNN model
        for epoch in range(max_epoch):
            for batch, D in enumerate(zip(train_dataloader,train_dataloader_old)):
                data, targets, index = D[0]
                data_old, _, index_old = D[1]
                data, targets, index = data.to(device), targets.to(device), index.to(device)
                data_old, index_old = data_old.to(device), index_old.to(device)
                
                optimizer.zero_grad()

                F = model(data)
                F_old = model(data_old)
                old_F_data = U_old[sample_index_old[index_old]]
                U[index, :] = F.data
                U_old_save[index, :] = F_old.data
                cnn_loss = criterion(F, F_old, old_F_data, B, S[index, :], sample_index[index])

                cnn_loss.backward()
                optimizer.step()

        # Update B
        expand_U = torch.zeros(B.shape).to(device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, gamma)

        # Total loss
        old_F_data = U_old[sample_index_old]
        iter_loss = calc_loss(U, U_old_save, old_F_data, B, S, code_length, sample_index, gamma, mu, ita, loss_type)
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
            topk2,
        )
        model.train()
        logger.info('map:{:.4f}'.format(mAP))
        hist_map.append(mAP)
        
        if mAP > best_mAP:
            best_mAP = mAP
            model.eval()
            print('save data for best map is %.4f of epoch %d'%(best_mAP,it+1))
            # Save checkpoints
            torch.save(query_code.cpu(), os.path.join('checkpoints', 'query_code2_down.t'))
            torch.save(B.cpu(), os.path.join('checkpoints', 'B_dataset2_down.t'))
            torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join('checkpoints', 'query_targets2.t'))
            torch.save(retrieval_targets.cpu(), os.path.join('checkpoints', 'database_targets2.t'))
            torch.save(model.cpu(), os.path.join('checkpoints', 'model_step2_down.t'))
            torch.save(best_mAP, os.path.join('checkpoints', 'best_map_down_data2.t'))
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
        topk2,
    )
    model.train()

    loss_plot(hist_loss,'./','data2_down_1')
    
    
    #construct dataset1 learned down code
    model = torch.load(os.path.join('checkpoints', 'model_step2_down.t'))
    best_mAP = torch.load(os.path.join('checkpoints', 'best_map_down_data2.t')) 
    model.cuda()
    
    print('load model best_map is %.4f'%best_mAP)
    retrieval_code_dataset1 = generate_code(model, retrieval_dataloader_old, code_length, device)
    torch.save(retrieval_code_dataset1.cpu(), os.path.join('checkpoints', 'U_database1_down_learned.t'))
    model.train()
    print('finish construct dataset1 down_code_learned,size is %d x %d'%(retrieval_code_dataset1.size(0), retrieval_code_dataset1.size(1)))
    
    
    query_code_dataset1 = generate_code(model, query_dataloader_old, code_length, device)
    map_dataset1 = evaluate.mean_average_precision(
        query_code_dataset1.to(device),
        retrieval_code_dataset1.to(device),
        query_dataloader_old.dataset.get_onehot_targets().to(device),
        retrieval_dataloader_old.dataset.get_onehot_targets().to(device),
        device,
        topk,
    )
    model.train()
    print('the map of dataset1 down_code_learned is %.4f'%map_dataset1)

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


def calc_loss(U, U_old_save, old_F_data, B, S, code_length, omega, gamma, mu, ita, loss_type):
    """
    Calculate loss.
    """
    #hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    #quantization_loss = ((U - B[omega, :]) ** 2).sum()
    #loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])
    
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    correlation_loss = (U @ torch.ones(U.shape[1], 1, device=U.device)).sum()
    replace_loss = ((code_length - (U_old_save * old_F_data).sum(1)) ** 2).sum()
    if loss_type == 'uidh':
        loss = (hash_loss + gamma * quantization_loss + mu * correlation_loss) / (U.shape[0] * B.shape[0]) + ita * replace_loss / (U_old_save.shape[0])
    elif loss_type == 'adsh':
        loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])
    else:
        raise ValueError('loss type error')
        
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



