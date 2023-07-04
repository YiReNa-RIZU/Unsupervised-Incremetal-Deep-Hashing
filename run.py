import torch
import argparse
import adsh
import os
import adsh_supervised
import adsh_unsupervised
import uidh_unsupervised
import uidh_supervised
import adsh_unsupervised_cifar10
import uidh_unsupervised_nuswide
import utils.evaluate as evaluate


from make_similarity_matrix import make_similarity_maxtrix
from loguru import logger
from data.data_loader import load_data
from draw_plot import mAP_plot
from adsh import generate_code
from show_images import show_topk_images
from cal_hashcode_num import cal_hashcode_num
from wrap_label import wrap_query_label, wrap_retrieval_label


def run():
    args = load_config()
    logger.add('logs/{time}.log', rotation='500 MB', level='INFO')
    logger.info(args)

    torch.backends.cudnn.benchmark = True
    
    
    
    # Load dataset
    if args.dataset == 'cifar-10':
        query_dataloader, _, retrieval_dataloader, MSM_dataloader = load_data(
            args.dataset,
            args.root,
            args.num_query,
            args.num_samples,
            args.batch_size,
            args.num_workers,
        )
    else: 
        query_dataloader, _, retrieval_dataloader = load_data(
            args.dataset,
            args.root,
            args.num_query,
            args.num_samples,
            args.batch_size,
            args.num_workers,
        )
    
    if args.dataset2 == 'cifar-10':
        query_dataloader2, _, retrieval_dataloader2, MSM_dataloader = load_data(
            args.dataset2,
            args.root2,
            args.num_query2,
            args.num_samples,
            args.batch_size,
            args.num_workers,
            )
    else:
        query_dataloader2, _, retrieval_dataloader2 = load_data(
            args.dataset2,
            args.root2,
            args.num_query2,
            args.num_samples,
            args.batch_size,
            args.num_workers,
            )
        
    
    match = [(0, 5), (1, 9), (2, 3), (3, 2), (4, 0), (5, 1), (6, 6), (7, 7), (8, 8), (9, 4)]
    torch.save(make_similarity_maxtrix('SCAN', MSM_dataloader, match),
                os.path.join('checkpoints','similarity_maxtrix.t'))
    del MSM_dataloader, match
    
    
    #train step1
    mAP_up, hist1 = adsh.train(
        query_dataloader,
        retrieval_dataloader,
        query_dataloader2,
        retrieval_dataloader2,
        args.code_length,
        args.device,
        args.lr,
        args.max_iter1,
        args.max_epoch,
        args.num_samples,
        args.batch_size,
        args.root,
        args.dataset,
        args.gamma,
        args.topk,
        args.topk2,
    )
    logger.info('[supervised code_length:{}][map:{:.4f}]'.format(args.code_length, mAP_up))
    
    if args.down_pre_mode == 'unsupervised' and (args.dataset == 'flickr25k' or args.dataset == 'nus-wide-tc21'):
        mAP_down, hist2 = adsh_unsupervised.train(
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.lr,
            args.max_iter2,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.dataset,
            args.gamma,
            args.topk,
        )
        logger.info('[unsupervised code_length:{}][map:{:.4f}]'.format(args.code_length, mAP_down))
        
        
    elif args.down_pre_mode == 'unsupervised' and args.dataset == 'cifar-10':
        mAP_down, hist2 = adsh_unsupervised_cifar10.train(
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.lr,
            args.max_iter2,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.dataset,
            args.gamma,
            args.topk,
        )
        logger.info('[unsupervised code_length:{}][map:{:.4f}]'.format(args.code_length, mAP_down))
    
    elif args.down_pre_mode == 'supervised': 
        mAP_down, hist2 = adsh_supervised.train(
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.lr,
            args.max_iter2,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.dataset,
            args.gamma,
            args.topk,
        )
        logger.info('[step1 down net supervised code_length:{}][map:{:.4f}]'.format(args.code_length, mAP_down))
        
        
    else:
        raise ValueError('up net mode error')
    
    
    
    B_data1_up = torch.load(os.path.join('checkpoints', 'B_dataset1_up.t'))
    B_data1_down = torch.load(os.path.join('checkpoints', 'B_dataset1_down.t'))
    
    Q_data1_up = torch.load(os.path.join('checkpoints', 'query_code1_up.t'))
    Q_data1_down = torch.load(os.path.join('checkpoints', 'query_code1_down.t'))
    
    B_data1_all = torch.cat([B_data1_up,B_data1_down],dim=1).to(args.device)
    Q_data1_all = torch.cat([Q_data1_up,Q_data1_down],dim=1).to(args.device)
    
    mAP_all = evaluate.mean_average_precision(
        Q_data1_all,
        B_data1_all,
        query_dataloader.dataset.get_onehot_targets().to(args.device),
        retrieval_dataloader.dataset.get_onehot_targets().to(args.device),
        args.device,
        args.topk,
    )
    logger.info('[ALL code_length:{}] on {} [map:{:.4f}]'.format(args.code_length*2, args.dataset, mAP_all))
    # mAP_plot(hist1,hist2,'./','data1_1')
    
    
    B_data2_up = torch.load(os.path.join('checkpoints', 'U_database2_up.t'))
    model_step1_down = torch.load(os.path.join('checkpoints', 'model_step1_down.t')).to(args.device)
    B_data2_down = generate_code(model_step1_down, retrieval_dataloader2, args.code_length, args.device)
    
    Q_data2_up = torch.load(os.path.join('checkpoints', 'query_code2_up.t'))
    Q_data2_down = generate_code(model_step1_down, query_dataloader2, args.code_length, args.device)
    
    B_data2_all = torch.cat([B_data2_up,B_data2_down],dim=1).to(args.device)
    Q_data2_all = torch.cat([Q_data2_up,Q_data2_down],dim=1).to(args.device)
    
    mAP_all2 = evaluate.mean_average_precision(
        Q_data2_all,
        B_data2_all,
        query_dataloader2.dataset.get_onehot_targets().to(args.device),
        retrieval_dataloader2.dataset.get_onehot_targets().to(args.device),
        args.device,
        args.topk2,
    )
    logger.info('[ALL code_length:{}] on {} [map:{:.4f}]'.format(args.code_length*2, args.dataset2, mAP_all2))

    
    retrieval_code, query_code = B_data1_all, Q_data1_all
    retrieval_code2, query_code2 = B_data2_all, Q_data2_all
    
    retrieval_code2 = torch.cat([retrieval_code, retrieval_code2], dim=0).to(args.device)
    query_code_all = torch.cat([query_code, query_code2], dim=0).to(args.device)
    
    retrieval_label = retrieval_dataloader.dataset.get_onehot_targets()
    retrieval_label2 = retrieval_dataloader2.dataset.get_onehot_targets()
    query_label = query_dataloader.dataset.get_onehot_targets()
    query_label2 = query_dataloader2.dataset.get_onehot_targets()

    
    Q = wrap_query_label(query_label, retrieval_label, query_label2, retrieval_label2, mode='all')
    R = wrap_retrieval_label(retrieval_label, retrieval_label2)
    print('q.size: ', Q.size())
    print('r.size: ', R.size())
    print(Q[0],R[0])
    
    mAP_new_all = evaluate.mean_average_precision(
        query_code_all,
        retrieval_code2,
        Q.to(args.device),
        R.to(args.device),
        args.device,
        -1,
    )
    print('mAP_new_all: ', mAP_new_all)
    
    
    Q = wrap_query_label(query_label, retrieval_label, query_label2, retrieval_label2, mode='d1')
    print('q.size: ', Q.size())
    print('r.size: ', R.size())
    print(Q[0],R[0])
    
    mAP_new_d1 = evaluate.mean_average_precision(
        query_code,
        retrieval_code2,
        Q.to(args.device),
        R.to(args.device),
        args.device,
        -1,
    )
    print('mAP_new_d1: ', mAP_new_d1)
    
    Q = wrap_query_label(query_label, retrieval_label, query_label2, retrieval_label2, mode='d2')
    print('q.size: ', Q.size())
    print('r.size: ', R.size())
    print(Q[0],R[0])
    
    mAP_new_d2 = evaluate.mean_average_precision(
        query_code2,
        retrieval_code2,
        Q.to(args.device),
        R.to(args.device),
        args.device,
        -1,
    )
    print('mAP_new_d2: ', mAP_new_d2)
    map_new_cal = (mAP_new_d1 + mAP_new_d2)/2
    print('map_new_cal:', map_new_cal)
    
    show_topk_images(query_dataloader, retrieval_dataloader, Q_data1_all,  B_data1_all, img_num=args.img_num, step='1')
    
    del B_data1_up, B_data1_down, B_data1_all,  Q_data1_up, Q_data1_down, Q_data1_all, retrieval_code, retrieval_code2
    del B_data2_up, B_data2_down, B_data2_all,  Q_data2_up, Q_data2_down, Q_data2_all, query_code, query_code2, query_code_all
    del Q, R, retrieval_label, retrieval_label2, query_label, query_label2
    
    
    
    #train step2
    if args.down_mode == 'unsupervised' and args.dataset2 == 'cifar-10':
        mAP_down_step2, hist3 = uidh_unsupervised.train(
            query_dataloader2,
            retrieval_dataloader2,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.lr,
            args.max_iter2,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.root2,
            args.dataset,
            args.dataset2,
            args.gamma,
            args.topk,
            args.topk2,
            args.loss_type,
            args.mu,
            args.ita,
        )
        
    elif args.down_mode == 'unsupervised' and (args.dataset2 == 'flickr25k' or args.dataset2 == 'nus-wide-tc21'):
        mAP_down_step2, hist3 = uidh_unsupervised_nuswide.train(
            query_dataloader2,
            retrieval_dataloader2,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.lr,
            args.max_iter2,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.root2,
            args.dataset,
            args.dataset2,
            args.gamma,
            args.topk,
            args.topk2,
            args.loss_type,
            args.mu,
            args.ita,
        )
    
    elif args.down_mode == 'supervised':
        mAP_down_step2, hist3 = uidh_supervised.train(
            query_dataloader2,
            retrieval_dataloader2,
            query_dataloader,
            retrieval_dataloader,
            args.code_length,
            args.device,
            args.lr,
            args.max_iter2,
            args.max_epoch,
            args.num_samples,
            args.batch_size,
            args.root,
            args.root2,
            args.dataset,
            args.dataset2,
            args.gamma,
            args.topk,
            args.topk2,
            args.loss_type,
            args.mu,
            args.ita,
        )
    
    else:
        raise ValueError('down net mode error')
    
    
    print(len(retrieval_dataloader.dataset))
    print(len(retrieval_dataloader2.dataset))
    print(len(query_dataloader.dataset))
    print(len(query_dataloader2.dataset))
    retrieval_code, query_code = cal_map(args.dataset, args.dataset, query_dataloader, retrieval_dataloader, args)
    retrieval_code2, query_code2 = cal_map(args.dataset, args.dataset2, query_dataloader2, retrieval_dataloader2, args)
    query_code, query_code2 = query_code.to(args.device), query_code2.to(args.device)
    
    retrieval_code2 = torch.cat([retrieval_code, retrieval_code2], dim=0).to(args.device)
    query_code_all = torch.cat([query_code, query_code2], dim=0).to(args.device)
    
    retrieval_label = retrieval_dataloader.dataset.get_onehot_targets()
    retrieval_label2 = retrieval_dataloader2.dataset.get_onehot_targets()
    query_label = query_dataloader.dataset.get_onehot_targets()
    query_label2 = query_dataloader2.dataset.get_onehot_targets()

    
    Q = wrap_query_label(query_label, retrieval_label, query_label2, retrieval_label2, mode='all')
    R = wrap_retrieval_label(retrieval_label, retrieval_label2)
    print('q.size: ', Q.size())
    print('r.size: ', R.size())
    print(Q[0],R[0])
    
    mAP_new_all = evaluate.mean_average_precision(
        query_code_all,
        retrieval_code2,
        Q.to(args.device),
        R.to(args.device),
        args.device,
        -1,
    )
    print('mAP_new_all: ', mAP_new_all)
    
    
    Q = wrap_query_label(query_label, retrieval_label, query_label2, retrieval_label2, mode='d1')
    print('q.size: ', Q.size())
    print('r.size: ', R.size())
    print(Q[0],R[0])
    
    mAP_new_d1 = evaluate.mean_average_precision(
        query_code,
        retrieval_code2,
        Q.to(args.device),
        R.to(args.device),
        args.device,
        -1,
    )
    print('mAP_new_d1: ', mAP_new_d1)
    
    Q = wrap_query_label(query_label, retrieval_label, query_label2, retrieval_label2, mode='d2')
    print('q.size: ', Q.size())
    print('r.size: ', R.size())
    print(Q[0],R[0])
    
    mAP_new_d2 = evaluate.mean_average_precision(
        query_code2,
        retrieval_code2,
        Q.to(args.device),
        R.to(args.device),
        args.device,
        -1,
    )
    print('mAP_new_d2: ', mAP_new_d2)
    map_new_cal = (mAP_new_d1 + mAP_new_d2)/2
    print('map_new_cal:', map_new_cal)
    
    
    
    retrieval_code = retrieval_code.to('cuda:0')
    query_code = query_code.to('cuda:0')
    show_topk_images(query_dataloader, retrieval_dataloader, query_code, retrieval_code, img_num=args.img_num, step='2')
    cal_hashcode_num()
    
  

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ADSH_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.',default='cifar-10')            ######
    parser.add_argument('--dataset2',
                        help='Dataset2 name.',default='nus-wide-tc21')               ######
    parser.add_argument('--root',
                        help='Path of dataset',default='../data-set')   ######
    parser.add_argument('--root2',
                        help='Path of dataset2',default='../data-set/nus-wide')  ######
    parser.add_argument('--down_pre_mode', 
                        help='mode of uidh step1 down net(supervised.unsupervised)',default='supervised',) 
    parser.add_argument('--down_mode', 
                        help='mode of uidh step2 down net(supervised.unsupervised)',default='unsupervised',)      ######
    parser.add_argument('--loss_type', 
                        help='step2 down net loss tpye(adsh.uidh)',default='uidh',)      ######
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--code-length', default=24, type=int,                              ######
                        help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max-iter1', default=50, type=int,
                        help='Number of iterations1.(default: 50)')
    parser.add_argument('--max-iter2', default=50, type=int,
                        help='Number of iterations2.(default: 50)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-query', default=10000, type=int,                              ######
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-query2', default=2100, type=int,                               ######
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=50000, type=int,                                   ######
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--topk2', default=5000, type=int,                                   ######
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default='0', type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--mu', default=50, type=float,
                    help='Hyper-parameter.(default: 50)')
    parser.add_argument('--ita', default=0.0, type=float,                                      ######
                    help='Hyper-parameter.(default: 1.0)')
    parser.add_argument('--img_num', default=20, type=int,                                      ######
                    help='sample of show images.(default: 3)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
        print('gpu is readly')

    # Hash code length
    #args.code_length = list(map(int, args.code_length.split(',')))

    return args

def cal_map(dataset1, dataset, query_dataloader, retrieval_dataloader, args):
    with torch.no_grad():
        model_up = torch.load(os.path.join('checkpoints', 'model_supervised.t'))
        model_down = torch.load(os.path.join('checkpoints', 'model_step2_down.t'))
        model_up, model_down = model_up.eval().to(args.device), model_down.eval().to(args.device)
        if dataset1 == dataset:
            B_up_code = torch.load(os.path.join('checkpoints', 'B_dataset1_up.t'))
            B_down_code = torch.load(os.path.join('checkpoints', 'U_database1_down_learned.t'))
            B = torch.cat([B_up_code,B_down_code], dim=1).to(args.device)
            topk = args.topk
        
        else:
            B_up_code = torch.load(os.path.join('checkpoints', 'U_database2_up.t'))
            B_down_code = torch.load(os.path.join('checkpoints', 'B_dataset2_down.t'))
            B = torch.cat([B_up_code,B_down_code], dim=1).to(args.device)
            topk = args.topk2
            
        query_up_code = generate_code(model_up, query_dataloader, args.code_length, args.device)
        query_down_code = generate_code(model_down, query_dataloader, args.code_length, args.device)
        query_code = torch.cat([query_up_code,query_down_code], dim=1)
        
        torch.save(B.cpu(), os.path.join('checkpoints', 'B_{}_step2_all_code.t').format(dataset))
        torch.save(query_code.cpu(), os.path.join('checkpoints', 'Q_{}_step2_all_code.t').format(dataset))
        
        mAP = evaluate.mean_average_precision(
            query_code.to(args.device),
            B,
            query_dataloader.dataset.get_onehot_targets().to(args.device),
            retrieval_dataloader.dataset.get_onehot_targets().to(args.device),
            args.device,
            topk,
        )
        print('dataset1 is %s ,map on %s is %.4f'%(dataset1, dataset, mAP))
        
    return B, query_code



if __name__ == '__main__':
    run()
