import torch
import torch.nn.functional as F
import numpy as np
from data_process.clinc150 import Bert_session, CLINC150_Bert
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from losses.triplet import TripletLoss, TripletLoss_avg
from losses.CenterTriplet import CenterTripletLoss
from losses.metrics import euclidean_dist
from models.Bert import Bert
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import time
import copy
import random


def compute_sample_weight(prototypes, sample_embdeddings):
    # prototypes:[80, embed_size]
    # sample_embed:[n*k, embed_size]
    # weight:[80, n*k]  只有当前训练集见过的类别有意义
    class_num = prototypes.size()[0]
    embedd_size = prototypes.size()[1]
    sample_num = sample_embdeddings.size()[0]
    theta = torch.std(sample_embdeddings)
    sample_embdeddings = sample_embdeddings.unsqueeze(0).repeat(150, 1, 1)
    prototypes = prototypes.unsqueeze(1).repeat(1, sample_num, 1)
    weight = torch.exp(
        -torch.pow(torch.norm(sample_embdeddings - prototypes, p=2, dim=2), exponent=2) / (2 * torch.pow(theta, 2)))
    return weight


def cl_memory_triplet_train(args):
    random.seed(0)
    with open(args.bert_path + 'word2id.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    train_dataset = CLINC150_Bert(args.data_path, args.bert_path, args.max_length, word2id, 'cl_train')
    # 用于生成每个session的训练数据和测试数据
    prototype_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    model = torch.load(args.model_save_path)
    fix_model = copy.deepcopy(model)
    model.to(device)
    if args.cl_optim == 'sgd':
        cl_optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
    elif args.cl_optim == 'adam':
        cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
    loss_fn = TripletLoss(args.margin)
    # 设置记忆存储器
    memory_size = args.class_num * args.K_shot
    memory = {'x1': torch.zeros([memory_size, args.max_length]).long(),
              'x2': torch.zeros([memory_size, args.max_length]).long(),
              'x3': torch.zeros([memory_size, args.max_length]).long(),
              'y': torch.zeros(memory_size).long()}
    # 存储base类别样本
    for cls in range(0, 70):
        for idx, sample_idx in enumerate(random.sample(range(0, 100), args.K_shot)):
            (memory['x1'][cls * args.K_shot + idx], memory['x2'][cls * args.K_shot + idx], memory['x3'][
                cls * args.K_shot + idx]), memory['y'][cls * args.K_shot + idx] = train_dataset.__getitem__(
                cls * 100 + sample_idx)
    # 计算原型
    model.eval()
    prototypes = torch.Tensor(args.class_num, args.encoder_dim).to(device)
    with torch.no_grad():
        for i, ((x1, x2, x3), y) in enumerate(prototype_loader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            prototypes[i] = model([x1, x2, x3]).mean(dim=0)
    fix_prototypes = copy.deepcopy(prototypes)
    # 计算第一个session的精度
    # session_data = Bert_session(args.data_path, word2id, args.max_length, 0, args.K_shot, 'cl_test', args.bert_path)
    # session_dataloader = DataLoader(session_data, batch_size=args.batch_size, shuffle=False)
    # correct = 0.0
    # with torch.no_grad():
    #     model.eval()
    #     for batch_idx, ((x1, x2, x3), y) in enumerate(session_dataloader):
    #         x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
    #         dist = euclidean_dist(model([x1, x2, x3]), prototypes[:70])
    #         _, pred = torch.max(-dist, dim=-1)
    #         correct += (pred == y).sum().float().item()
    #     first_acc = round(correct / len(session_data), 4)
    all_result = []
    for times in range(args.test_times):
        start_time = time.time()
        session_loss = [0.0] * 9
        session_acc = [0.0] * 9
        session_acc[0] = 0.9733
        session_data = Bert_session(args.data_path, word2id, args.max_length, 0, args.K_shot, 'cl_test', args.bert_path)
        model = copy.deepcopy(fix_model)
        model = model.to(device)
        if args.cl_optim == 'sgd':
            cl_optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        elif args.cl_optim == 'adam':
            cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        # 冻结bert除6层前参数 并查看可训练层
        for name, para in model.named_parameters():
            if '6' not in name:
                para.requires_grad = False
            else:
                break
        # for para in model.parameters():
        #     para.requires_grad = True
        prototypes = copy.deepcopy(fix_prototypes)
        print('times:', times)
        for i in range(1, 9):
            every_session_acc = [0.0] * 9
            # 增量学习
            print('#', i, end=' ')
            session_data.set_session(i)
            session_dataloader = DataLoader(session_data, batch_size=args.batch_size, shuffle=False)
            if args.random_test:
                (x1, x2, x3), y = session_data.get_random_train_session()
            else:
                (x1, x2, x3), y = session_data.get_train_session()
            # 存储当前session数据
            memory_idx = (i - 1) * args.N_way * args.K_shot + 700
            memory['x1'][memory_idx:args.N_way * args.K_shot + memory_idx], memory['x2'][
                                                                            memory_idx:args.N_way * args.K_shot + memory_idx], \
            memory['x3'][memory_idx:args.N_way * args.K_shot + memory_idx], memory['y'][
                                                                            memory_idx:args.N_way * args.K_shot + memory_idx] = x1, x2, x3, y
            ###
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            if args.regularization == 'lwf':
                old_model = copy.deepcopy(model)
                old_model.eval()
                with torch.no_grad():
                    for paras in old_model.parameters():
                        paras.requires_grad = False
                    old_out_put = old_model([x1, x2, x3])
            if args.regularization == 'nl2':
                old_model = copy.deepcopy(model)
                old_model.eval()
                for paras in old_model.parameters():
                    paras.requires_grad = False
            if args.sdc != 'false':
                old_model = copy.deepcopy(model)
                old_model.eval()
            model.train()
            for j in range(args.session_epoch):
                cl_optimizer.zero_grad()
                logits = model([x1, x2, x3])
                loss = loss_fn(logits, y)
                if args.regularization == 'lwf':
                    loss += torch.norm(logits - old_out_put, dim=1).sum()
                if args.regularization == 'nl2':
                    for key in model.state_dict().keys():
                        loss += args.r_xishu * (
                            torch.pow(model.state_dict()[key] - old_model.state_dict()[key], exponent=2).sum())
                loss.backward()
                cl_optimizer.step()
                session_loss[i] += loss.item()
            # 使用至今的所有类重放
            if args.replay:
                for k in range(args.replay_epoch):
                    old_x1, old_x2, old_x3, old_y = torch.zeros(
                        [args.replay_batch, args.max_length]).long(), torch.zeros(
                        [args.replay_batch, args.max_length]).long(), torch.zeros(
                        [args.replay_batch, args.max_length]).long(), torch.zeros(args.replay_batch).long()
                    for idx, sample_idx in enumerate(
                            random.sample(range(0, memory_idx), args.replay_batch - 30)):
                        old_x1[idx], old_x2[idx], old_x3[idx], old_y[idx] = memory['x1'][sample_idx], memory['x2'][
                            sample_idx], memory['x3'][sample_idx], memory['y'][sample_idx]
                    for idx, sample_idx in enumerate(
                            random.sample(range(memory_idx, memory_idx + args.K_shot * args.N_way), 30)):
                        old_x1[idx + args.replay_batch - 30], old_x2[idx + args.replay_batch - 30], old_x3[
                            idx + args.replay_batch - 30], old_y[idx + args.replay_batch - 30] = memory['x1'][
                                                                                                     sample_idx], \
                                                                                                 memory['x2'][
                                                                                                     sample_idx], \
                                                                                                 memory['x3'][
                                                                                                     sample_idx], \
                                                                                                 memory['y'][sample_idx]

                    old_x1, old_x2, old_x3, old_y = old_x1.to(device), old_x2.to(device), old_x3.to(device), old_y.to(
                        device)
                    cl_optimizer.zero_grad()
                    logits = model([old_x1, old_x2, old_x3])
                    loss = loss_fn(logits, old_y)
                    loss.backward()
                    cl_optimizer.step()
            session_loss[i] = round(session_loss[i] / (args.session_epoch + 0.0001), 4)
            # 在所有遇见过的类上测试
            correct = 0
            with torch.no_grad():
                model.eval()
                out_put = model([x1, x2, x3])
                # 语义漂移补偿
                if args.sdc == 'all':
                    old_memory_output = old_model(
                        [memory['x1'].to(device), memory['x2'].to(device), memory['x3'].to(device)])
                    new_memory_output = model(
                        [memory['x1'].to(device), memory['x2'].to(device), memory['x3'].to(device)])
                    weight = compute_sample_weight(prototypes, old_memory_output)
                    for cls in range(0, 70 + (i - 1) * args.N_way):
                        compensation = (
                                (new_memory_output - old_memory_output) * weight[cls].reshape(memory['x1'].size()[0],
                                                                                              1) / weight[
                                    cls].sum()).sum(dim=0)
                        prototypes[cls] = prototypes[cls] + compensation
                elif args.sdc == 'base':
                    old_memory_output = old_model(
                        [memory['x1'].to(device), memory['x2'].to(device), memory['x3'].to(device)])
                    new_memory_output = model(
                        [memory['x1'].to(device), memory['x2'].to(device), memory['x3'].to(device)])
                    weight = compute_sample_weight(prototypes, old_memory_output)
                    for cls in range(0, 70):
                        compensation = (
                                (new_memory_output - old_memory_output) * weight[cls].reshape(memory['x1'].size()[0],
                                                                                              1) / weight[
                                    cls].sum()).sum(dim=0)
                        prototypes[cls] = prototypes[cls] + compensation
                # 重新计算类原型
                if args.recompute_proto == 'all':
                    for cls in range(0, 70 + (i - 1) * args.N_way):
                        old_x1, old_x2, old_x3 = memory['x1'][cls * args.K_shot:(cls + 1) * args.K_shot], memory['x2'][
                                                                                                          cls * args.K_shot:(
                                                                                                                                    cls + 1) * args.K_shot], \
                                                 memory['x3'][cls * args.K_shot:(cls + 1) * args.K_shot]
                        old_x1, old_x2, old_x3 = old_x1.to(device), old_x2.to(device), old_x3.to(device)
                        prototypes[cls] = model([old_x1, old_x2, old_x3]).mean(dim=0)
                elif args.recompute_proto == 'new':
                    for cls in range(70, 70 + (i - 1) * args.N_way):
                        old_x1, old_x2, old_x3 = memory['x1'][cls * args.K_shot:(cls + 1) * args.K_shot], memory['x2'][
                                                                                                          cls * args.K_shot:(
                                                                                                                                    cls + 1) * args.K_shot], \
                                                 memory['x3'][cls * args.K_shot:(cls + 1) * args.K_shot]
                        old_x1, old_x2, old_x3 = old_x1.to(device), old_x2.to(device), old_x3.to(device)
                        prototypes[cls] = model([old_x1, old_x2, old_x3]).mean(dim=0)
                # 得到新的类的原型
                out_put = out_put.view(10, args.K_shot, args.encoder_dim)
                new_class_proto = out_put.mean(dim=1)
                prototypes[i * 10 + 60:i * 10 + 70] = new_class_proto
                for batch_idx, ((x1, x2, x3), y) in enumerate(session_dataloader):
                    x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                    dist = euclidean_dist(model([x1, x2, x3]), prototypes[:i * 10 + 70])
                    _, pred = torch.max(-dist, dim=-1)
                    correct += (pred == y).sum().float().item()
                    correct_num = (pred == y).sum().float().item()
                    if batch_idx < 7:
                        every_session_acc[0] += correct_num
                    elif batch_idx < 8:
                        every_session_acc[1] += correct_num
                    elif batch_idx < 9:
                        every_session_acc[2] += correct_num
                    elif batch_idx < 10:
                        every_session_acc[3] += correct_num
                    elif batch_idx < 11:
                        every_session_acc[4] += correct_num
                    elif batch_idx < 12:
                        every_session_acc[5] += correct_num
                    elif batch_idx < 13:
                        every_session_acc[6] += correct_num
                    elif batch_idx < 14:
                        every_session_acc[7] += correct_num
                    elif batch_idx < 15:
                        every_session_acc[8] += correct_num
                session_acc[i] = round(correct / len(session_data), 4)
                every_session_acc[0] = round(every_session_acc[0] / (len(session_data.all_data) * 7 / 15), 3)
                for idx in range(1, 9):
                    every_session_acc[idx] = round(every_session_acc[idx] / (len(session_data.all_data) / 15), 3)
                # print(' ', every_session_acc[0], ' ', every_session_acc[i], ' ', every_session_acc)
                # print('session:{} val_acc:{:.4f}'.format(i, session_acc[i]))
        print('\nacc:', session_acc)
        # print('loss:', session_loss)
        # print('以上结果超参配置：', args.session_epoch, ' ', args.regularization, ' ', args.cl_lr, ' ', args.cl_optim)
        all_result.append(session_acc)
        end_time = time.time()
        # print('用时：', (end_time - start_time) / 60, '分')
    return all_result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bert_path', type=str, default='D:/code/pretrained_model/bert/uncased_L-12_H-768_A-12/')
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='D:/code/dataset/CLINC150/data_full.json')
    argparser.add_argument('--max_length', help='轮次', type=int,
                           default=28)
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=256)
    argparser.add_argument('--class_num', help='类别数量', type=int,
                           default=150)
    argparser.add_argument('--lr', help='学习率', type=float,
                           default=0.1)

    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--encoder_dim', type=int, default=768)
    argparser.add_argument('--losses', help='loss function', type=str, default='triplet')
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='clinc_result/clinc_base_bert_ft6_triplet_m0.5_nodecay_adam3e-4.txt')
    argparser.add_argument('--K_shot', help='每类样本数量', type=int, default=10)
    argparser.add_argument('--N_way', help='每个task类别数量', type=int, default=10)
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='clinc_result/clinc_base_bert_ft6_triplet_m0.5_nodecay_adam3e-4.pth')
    argparser.add_argument('--weight_decay', help='', type=float, default=0.0)
    argparser.add_argument('--reuse', help='是否使用训练过的模型', type=bool, default=True)
    argparser.add_argument('--cl_lr', help='增量学习学习率', type=float,
                           default=1e-5)
    argparser.add_argument('--cl_optim', help='使用哪种优化器', type=str, default='adam')
    argparser.add_argument('--regularization', help='使用什么正则化 比如lwf', type=str, default='nolwf')
    argparser.add_argument('--session_epoch', help='增量任务轮次', type=int, default=0)
    argparser.add_argument('--replay_epoch', help='重放任务轮次', type=int, default=50)
    argparser.add_argument('--replay_batch', help='重放批次大小', type=int, default=256)
    argparser.add_argument('--replay', help='是否重放', type=bool, default=True)
    # 可选参数为all、new、false
    argparser.add_argument('--recompute_proto', help='是否重计算原型', type=str, default='new')
    # 可选参数为all、base、false
    argparser.add_argument('--sdc', help='是否补偿语义漂移', type=str, default='base')
    argparser.add_argument('--random_test', help='是否随机抽取session数据', type=bool, default=True)
    argparser.add_argument('--r_xishu', help='正则化系数', type=float, default=0.05)
    argparser.add_argument('--test_times', help='测试次数', type=int, default=10)
    args = argparser.parse_args()
    all_result = cl_memory_triplet_train(args=args)
    print('梯度下降次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：', args.r_xishu, ' 学习率：', args.cl_lr,
          ' 优化器：', args.cl_optim, 'K_shot:', args.K_shot)
    # print(all_result)
    print('avg result:')
    print(torch.Tensor(all_result).mean(dim=0))
    print(args)
    torch.cuda.empty_cache()
    print('**************************************************************************')
    print('**************************************************************************')
    print('**************************************************************************')
    ########
    # args.recompute_proto = 'new'
    # args.sdc = 'base'
    # all_result = cl_memory_triplet_train(args=args)
    # print('avg result:')
    # print(torch.Tensor(all_result).mean(dim=0))
    # print(args)
    # print('**************************************************************************')
    # print('**************************************************************************')
    # print('**************************************************************************')
    # torch.cuda.empty_cache()
    # ###############
    # args.recompute_proto = 'all'
    # args.sdc = 'false'
    # all_result = cl_memory_triplet_train(args=args)
    # print('avg result:')
    # print(torch.Tensor(all_result).mean(dim=0))
    # print(args)
    # print('**************************************************************************')
    # print('**************************************************************************')
    # print('**************************************************************************')
    # torch.cuda.empty_cache()
    # ###############
    # args.recompute_proto = 'false'
    # args.sdc = 'all'
    # all_result = cl_memory_triplet_train(args=args)
    # print('avg result:')
    # print(torch.Tensor(all_result).mean(dim=0))
    # print(args)
    # print('**************************************************************************')
    # print('**************************************************************************')
    # print('**************************************************************************')
    # torch.cuda.empty_cache()
