# 10/7
import os
import torch.backends.cudnn as cudnn
from config import ARGS
import util
from dataset.dataset_user_sep import UserSepDataset
from network.DKT import DKT
from network.DKVMN import DKVMN
from network.NPA import NPA
from network.SAKT import SAKT
from constant import QUESTION_NUM
from trainer import Trainer
import numpy as np

import time
from torch.utils import data
from tqdm import tqdm
from itertools import repeat, chain, islice
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data.distributed

from network.util_network import ScheduledOptim, NoamOpt
from torch.autograd import Variable

def get_model():
    '''
    if ARGS.model == 'DKT':
        model = DKT(ARGS.input_dim, ARGS.hidden_dim, ARGS.num_layers, QUESTION_NUM[ARGS.dataset_name],
                    ARGS.dropout).to(ARGS.device)
        d_model = ARGS.hidden_dim

    elif ARGS.model == 'DKVMN':
        model = DKVMN(ARGS.key_dim, ARGS.value_dim, ARGS.summary_dim, QUESTION_NUM[ARGS.dataset_name],
                      ARGS.concept_num).to(ARGS.device)
        d_model = ARGS.value_dim

    elif ARGS.model == 'NPA':
        model = NPA(ARGS.input_dim, ARGS.hidden_dim, ARGS.attention_dim, ARGS.fc_dim,
                    ARGS.num_layers, QUESTION_NUM[ARGS.dataset_name], ARGS.dropout).to(ARGS.device)
        d_model = ARGS.hidden_dim
    '''
    if ARGS.model == 'SAKT':
        model = SAKT(ARGS.hidden_dim, QUESTION_NUM[ARGS.dataset_name], ARGS.num_layers,
                     ARGS.num_head, ARGS.dropout)
        model.load_state_dict(torch.load(ARGS.weight_path + '265000.pt'))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)

        model.to(ARGS.device)
        d_model = ARGS.hidden_dim

    else:
        raise NotImplementedError

    return model, d_model

best_acc1 = 0

def run(i):
    global best_acc1

    user_base_path = '{}/{}/processed'.format(ARGS.base_path, ARGS.dataset_name)

    train_data_path = '{}/{}/train/'.format(user_base_path, i)
    val_data_path = '{}/{}/val/'.format(user_base_path, i)
    test_data_path = '{}/{}/test/'.format(user_base_path, i)

    cudnn.benchmark = True
    print('Run...')
    print(train_data_path) 

    train_sample_infos, num_of_train_user = util.get_data_user_sep(train_data_path)
    val_sample_infos, num_of_val_user = util.get_data_user_sep(val_data_path)
    test_sample_infos, num_of_test_user = util.get_data_user_sep(test_data_path)

    print('End reading...')
    
    # Tensor 반환
#     train_data = UserSepDataset('train', train_sample_infos, ARGS.dataset_name)
#     val_data = UserSepDataset('val', val_sample_infos, ARGS.dataset_name)
#     test_data = UserSepDataset('test', test_sample_infos, ARGS.dataset_name)

    print('Train: # of users: {}, # of samples: {}'.format(num_of_train_user, len(train_sample_infos)))
    print('Validation: # of users: {}, # of samples: {}'.format(num_of_val_user, len(val_sample_infos)))
    print('Test: # of users: {}, # of samples: {}'.format(num_of_test_user, len(test_sample_infos)))

    model, d_model = get_model()

    criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()
    adam = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
    optimizer =  NoamOpt(
            model_size=d_model, factor=1, warmup=ARGS.warm_up_step_count, optimizer=adam)

    # Train #######################################################
    # 일단 train & validate만!
    print('Start Train')
    
    train_gen = data.DataLoader(
        dataset=UserSepDataset('train', train_sample_infos, ARGS.dataset_name), shuffle=True,
        batch_size=ARGS.train_batch,
        num_workers=ARGS.num_workers, pin_memory=True)
    val_gen = data.DataLoader(
        dataset=UserSepDataset('val', val_sample_infos, ARGS.dataset_name), shuffle=False,
        batch_size=ARGS.test_batch, num_workers=ARGS.num_workers,
        pin_memory=True)
    test_gen = data.DataLoader(
        dataset=test_data, shuffle=False,
        batch_size=ARGS.test_batch, pin_memory = True, num_workers=ARGS.num_workers)


    # epoch 마다 반복
    for epoch in range(0, ARGS.num_epochs):
        train(train_gen, val_gen, model, criterion, optimizer, epoch)
        
    print('[Best acc] : {}'.format(best_acc1))

    #best_acc1 = test(test_gen)
    return best_acc1

def train(train_gen, val_gen, model, criterion, optimizer, epoch):
    model.train()
    num_corrects = 0
    num_total = 0
    
    end = time.time()
    length = len(train_gen)
    losses = 0
      
    for i, batch in enumerate(train_gen):
        i += 1
        start_time = time.time()
        #data_time.update(time.time() - end)

        if i%1000 == 0:
            print('{i} is completed.'.format(i=i))
        
        batch = {k: t.to(ARGS.device, non_blocking=True) for k, t in batch.items()}
        output = model(batch['input'], batch['target_id'])
        
        loss = criterion(output, batch['label'].float())
        losses += loss.mean().item()
       
        # num_corrects += torch.sum((torch.sigmoid(output).to(ARGS.device, non_blocking=True) >= 0.5).long() == batch['label'])
        # num_total += batch['label'].size(0)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        
        # measure elapsed time)
        end = time.time()

        if i % 100000 == 0:
            #acc = num_corrects.double()/num_total
            print('[Train {i}/{l} Loss] :{loss:.3f}, [Train time] :{t}'.
                format(i=i, l=len(train_gen), 
                loss=losses/ARGS.print_freq,
                t=time.time()-start_time))

            losses = 0
            
            validate(val_gen, model, criterion)
            print('')
       
def validate(val_gen, model, criterion):
    global best_acc1

    num_corrects = 0
    num_total = 0
    losses = 0

    start_time = time.time()
    model.eval()

    with torch.no_grad():
        end = time.time()
        i = 0
        
        for batch in val_gen:
            i += 1
            batch = {k: t.to(ARGS.device, non_blocking=True) for k, t in batch.items()}
            output = model(batch['input'], batch['target_id'])

            loss = criterion(output, batch['label'].float())
            losses += loss.mean().item()

            num_corrects += torch.sum((torch.sigmoid(output).to(ARGS.device, non_blocking=True) >= 0.5).long() == batch['label'])
            num_total += batch['label'].size(0)

            end = time.time()

            if i % ARGS.print_freq == 0:
                acc = num_corrects / num_total
                print('acc', acc.item())
                print('[Valid {i}/{l} acc] :{acc:.4f}, [Loss] :{loss:.3f}, [Train time] :{t}'.
                    format(i=i, l=len(val_gen), 
                    acc=acc.item(), 
                    loss=losses/ARGS.print_freq,
                    t=time.time()-start_time))

                losses = 0

                if acc > best_acc1:
                    best_acc1 = acc
                    cur_weight = model.state_dict()
                    torch.save(cur_weight, '{}{}.pt'.format(ARGS.weight_path, i))

if __name__ == '__main__':

    if ARGS.cross_validation is False:
        max_acc = run(1)
        print('[MAX_ACC]', max_acc)
    else:
        acc_list = []
        auc_list = []

        for i in range(1, 6):
            print('{}th dataset'.format(i))
            test_acc, test_auc = run(i)
            acc_list.append(test_acc)
            auc_list.append(test_auc)

        acc_array = np.asarray(acc_list)
        auc_array = np.asarray(auc_list)
        print('mean acc: {}, auc: {}'.format(np.mean(acc_array), np.mean(auc_array)))
        print('std acc: {}, auc: {}'.format(np.std(acc_array), np.std(auc_array)))
