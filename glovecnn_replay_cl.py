import torch
import torch.nn.functional as F
import numpy as np
from data_process.fewrel import Joint_dataset, FewRel_Bert, Bert_Session, Session
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
    sample_embdeddings = sample_embdeddings.unsqueeze(0).repeat(80, 1, 1)
    prototypes = prototypes.unsqueeze(1).repeat(1, sample_num, 1)
    weight = torch.exp(
        -torch.pow(torch.norm(sample_embdeddings - prototypes, p=2, dim=2), exponent=2) / (2 * torch.pow(theta, 2)))
    return weight


def cl_memory_triplet_train(args):
    random.seed(0)
    with open(args.word2id_path) as f:
        word2id = json.load(f)
    train_dataset = Joint_dataset(args.data_path, word2id, max_length=36, mode='cl_train')
    prototype_loader = DataLoader(train_dataset, batch_size=560, shuffle=False)
    # 用于生成每个session的训练数据和测试数据
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
              'pos1': torch.zeros([memory_size, args.max_length]).long(),
              'pos2': torch.zeros([memory_size, args.max_length]).long(),
              'y': torch.zeros(memory_size).long()}
    # 存储base类别样本
    for cls in range(0, 40):
        for idx, sample_idx in enumerate(random.sample(range(0, 560), args.K_shot)):
            (memory['x1'][cls * args.K_shot + idx], memory['pos1'][cls * args.K_shot + idx],
             memory['pos2'][cls * args.K_shot + idx], _), memory['y'][
                cls * args.K_shot + idx] = train_dataset.__getitem__(
                cls * 560 + sample_idx)
    # 计算原型
    model.eval()
    prototypes = torch.Tensor(args.class_num, args.encoder_dim).to(device)
    with torch.no_grad():
        for i, ((x1, pos1, pos2, _), y) in enumerate(prototype_loader):
            x1, pos1, pos2, y = x1.to(device), pos1.to(device), pos2.to(
                device), y.to(device)
            prototypes[i] = model(x1, pos1, pos2).mean(dim=0)
    fix_prototypes = copy.deepcopy(prototypes)
    # 计算第一个session的精度
    session_data = Session(args.data_path, word2id, 36, session=0, K=args.K_shot, mode='cl_test')
    session_dataloader = DataLoader(session_data, batch_size=args.batch_size, shuffle=False)
    correct = 0.0
    with torch.no_grad():
        model.eval()
        for batch_idx, ((x1, pos1, pos2, _), y) in enumerate(session_dataloader):
            x1, pos1, pos2, y = x1.to(device), pos1.to(device), pos2.to(
                device), y.to(device)
            dist = euclidean_dist(model(x1, pos1, pos2), prototypes[:40])
            _, pred = torch.max(-dist, dim=-1)
            correct += (pred == y).sum().float().item()
        first_acc = round(correct / len(session_data), 4)
    all_result = []
    for times in range(args.test_times):
        start_time = time.time()
        session_loss = [0.0] * 9
        session_acc = [0.0] * 9
        session_acc[0] = first_acc
        session_data = Session(args.data_path, word2id, 36, session=1, K=args.K_shot, mode='cl_test')
        model = copy.deepcopy(fix_model)
        model = model.to(device)
        if args.cl_optim == 'sgd':
            cl_optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        elif args.cl_optim == 'adam':
            cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        prototypes = copy.deepcopy(fix_prototypes)
        print('times:', times)
        for i in range(1, 9):
            every_session_acc = [0.0] * 9
            # 增量学习
            print('#', i, end=' ')
            session_data.set_session(i)
            session_dataloader = DataLoader(session_data, batch_size=args.batch_size, shuffle=False)
            if args.random_test:
                (x1, pos1, pos2, _), y = session_data.get_random_train_session()
            else:
                (x1, pos1, pos2, _), y = session_data.get_train_session()
            # 存储当前session数据
            memory_idx = (i - 1) * args.N_way * args.K_shot + 40 * args.K_shot
            memory['x1'][memory_idx:args.N_way * args.K_shot + memory_idx], memory['pos1'][
                                                                            memory_idx:args.N_way * args.K_shot + memory_idx], \
            memory['pos2'][memory_idx:args.N_way * args.K_shot + memory_idx], memory['y'][
                                                                              memory_idx:args.N_way * args.K_shot + memory_idx] = x1, pos1, pos2, y
            ###
            x1, pos1, pos2, y = x1.to(device), pos1.to(device), pos2.to(device), y.to(device)
            if args.regularization == 'lwf':
                old_model = copy.deepcopy(model)
                old_model.eval()
                with torch.no_grad():
                    for paras in old_model.parameters():
                        paras.requires_grad = False
                    old_out_put = old_model(x1, pos1, pos2)
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
                logits = model(x1, pos1, pos2)
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
                new_num = args.replay_batch_newtask
                for k in range(args.replay_epoch):
                    old_x1, old_pos1, old_pos2, old_y = torch.zeros(
                        [args.replay_batch, args.max_length]).long(), torch.zeros(
                        [args.replay_batch, args.max_length]).long(), torch.zeros(
                        [args.replay_batch, args.max_length]).long(), torch.zeros(args.replay_batch).long()
                    for idx, sample_idx in enumerate(
                            random.sample(range(0, memory_idx), args.replay_batch - new_num)):
                        old_x1[idx], old_pos1[idx], old_pos2[idx], old_y[idx] = memory['x1'][
                                                                                    sample_idx], \
                                                                                memory[
                                                                                    'pos1'][
                                                                                    sample_idx], \
                                                                                memory[
                                                                                    'pos2'][
                                                                                    sample_idx], \
                                                                                memory['y'][
                                                                                    sample_idx]
                    for idx, sample_idx in enumerate(
                            random.sample(range(memory_idx, memory_idx + args.K_shot * args.N_way), new_num)):
                        old_x1[idx + args.replay_batch - new_num], old_pos1[idx + args.replay_batch - new_num], old_pos2[
                            idx + args.replay_batch - new_num], old_y[idx + args.replay_batch - new_num] = memory['x1'][
                                                                                                     sample_idx], \
                                                                                                 memory[
                                                                                                     'pos1'][
                                                                                                     sample_idx], \
                                                                                                 memory[
                                                                                                     'pos2'][
                                                                                                     sample_idx], \
                                                                                                 memory['y'][
                                                                                                     sample_idx]

                    old_x1, old_pos1, old_pos2, old_y = old_x1.to(device), old_pos1.to(device), old_pos2.to(
                        device), old_y.to(
                        device)
                    cl_optimizer.zero_grad()
                    logits = model(old_x1, old_pos1, old_pos2)
                    loss = loss_fn(logits, old_y)
                    loss.backward()
                    cl_optimizer.step()
            session_loss[i] = round(session_loss[i] / (args.session_epoch + 0.0001), 4)
            # 在所有遇见过的类上测试
            correct = 0
            with torch.no_grad():
                model.eval()
                out_put = model(x1, pos1, pos2)
                # 语义漂移补偿
                if args.sdc == 'all':
                    old_memory_output = old_model(memory['x1'].to(device), memory['pos1'].to(device),
                                                  memory['pos2'].to(device))
                    new_memory_output = model(memory['x1'].to(device), memory['pos1'].to(device),
                                              memory['pos2'].to(device))
                    weight = compute_sample_weight(prototypes, old_memory_output)
                    for cls in range(0, 40 + (i - 1) * args.N_way):
                        compensation = (
                                (new_memory_output - old_memory_output) * weight[cls].reshape(memory['x1'].size()[0],
                                                                                              1) / weight[
                                    cls].sum()).sum(dim=0)
                        prototypes[cls] = prototypes[cls] + compensation
                elif args.sdc == 'base':
                    old_memory_output = old_model(memory['x1'].to(device), memory['pos1'].to(device),
                                                  memory['pos2'].to(device))
                    new_memory_output = model(memory['x1'].to(device), memory['pos1'].to(device),
                                              memory['pos2'].to(device))
                    weight = compute_sample_weight(prototypes, old_memory_output)
                    for cls in range(0, 40):
                        compensation = (
                                (new_memory_output - old_memory_output) * weight[cls].reshape(memory['x1'].size()[0],
                                                                                              1) / weight[
                                    cls].sum()).sum(dim=0)
                        prototypes[cls] = prototypes[cls] + compensation
                # 重新计算类原型
                if args.recompute_proto == 'all':
                    for cls in range(0, 40 + (i - 1) * args.N_way):
                        old_x1, old_pos1, old_pos2 = memory['x1'][
                                                     cls * args.K_shot:(cls + 1) * args.K_shot], memory['pos1'][
                                                                                                 cls * args.K_shot:(
                                                                                                                           cls + 1) * args.K_shot], \
                                                     memory['pos2'][cls * args.K_shot:(cls + 1) * args.K_shot]
                        old_x1, old_pos1, old_pos2 = old_x1.to(device), old_pos1.to(device), old_pos2.to(device)
                        prototypes[cls] = model(old_x1, old_pos1, old_pos2).mean(dim=0)
                elif args.recompute_proto == 'new':
                    for cls in range(40, 40 + (i - 1) * args.N_way):
                        old_x1, old_pos1, old_pos2 = memory['x1'][
                                                     cls * args.K_shot:(cls + 1) * args.K_shot], memory['pos1'][
                                                                                                 cls * args.K_shot:(
                                                                                                                           cls + 1) * args.K_shot], \
                                                     memory['pos2'][cls * args.K_shot:(cls + 1) * args.K_shot]
                        old_x1, old_pos1, old_pos2 = old_x1.to(device), old_pos1.to(device), old_pos2.to(device)
                        prototypes[cls] = model(old_x1, old_pos1, old_pos2).mean(dim=0)
                # 得到新的类的原型
                out_put = out_put.view(args.N_way, args.K_shot, args.encoder_dim)
                new_class_proto = out_put.mean(dim=1)
                prototypes[i * args.N_way + 35:i * args.N_way + 40] = new_class_proto
                for batch_idx, ((x1, pos1, pos2, _), y) in enumerate(session_dataloader):
                    x1, pos1, pos2, y = x1.to(device), pos1.to(device), pos2.to(
                        device), y.to(device)
                    dist = euclidean_dist(model(x1, pos1, pos2), prototypes[:i * args.N_way + 40])
                    _, pred = torch.max(-dist, dim=-1)
                    correct += (pred == y).sum().float().item()
                    correct_num = (pred == y).sum().float().item()
                    if batch_idx < 40:
                        every_session_acc[0] += correct_num
                    elif batch_idx < 45:
                        every_session_acc[1] += correct_num
                    elif batch_idx < 50:
                        every_session_acc[2] += correct_num
                    elif batch_idx < 55:
                        every_session_acc[3] += correct_num
                    elif batch_idx < 60:
                        every_session_acc[4] += correct_num
                    elif batch_idx < 65:
                        every_session_acc[5] += correct_num
                    elif batch_idx < 70:
                        every_session_acc[6] += correct_num
                    elif batch_idx < 75:
                        every_session_acc[7] += correct_num
                    elif batch_idx < 80:
                        every_session_acc[8] += correct_num
                session_acc[i] = round(correct / len(session_data), 4)
                every_session_acc[0] = round(every_session_acc[0] / 3200, 3)
                for idx in range(1, 9):
                    every_session_acc[idx] = round(every_session_acc[idx] / 400, 3)
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
    argparser.add_argument('--glove_path', type=str, default='D:/code/pretrained_model/glove.6B')
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='data/all_wiki.json')
    argparser.add_argument('--word2id_path', type=str,
                           default='data/word2id.json')
    argparser.add_argument('--max_length', help='轮次', type=int,
                           default=36)
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=80)
    argparser.add_argument('--class_num', help='类别数量', type=int,
                           default=80)
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=1)
    argparser.add_argument('--encoder_dim', type=int, default=512)
    argparser.add_argument('--losses', help='loss function', type=str, default='triplet')
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='result/.txt')
    argparser.add_argument('--K_shot', help='每类样本数量', type=int, default=10)
    argparser.add_argument('--N_way', help='每个task类别数量', type=int, default=5)
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='result/base_glove_triplet_m1_adam1e-3_encoder512_pos_nodecay.pth')  # base_bert_ft6_cnn_pos_triplet base_bertcnn_pos_triplet_adam2e-5
    argparser.add_argument('--weight_decay', help='', type=float, default=0.0)
    argparser.add_argument('--reuse', help='是否使用训练过的模型', type=bool, default=True)
    argparser.add_argument('--cl_lr', help='增量学习学习率', type=float,
                           default=1e-4)
    argparser.add_argument('--cl_optim', help='使用哪种优化器', type=str, default='adam')
    argparser.add_argument('--regularization', help='使用什么正则化 比如lwf', type=str, default='nolwf')
    argparser.add_argument('--session_epoch', help='增量任务轮次', type=int, default=0)
    argparser.add_argument('--replay_epoch', help='重放任务轮次', type=int, default=60)
    argparser.add_argument('--replay_batch', help='重放批次大小', type=int, default=150)
    argparser.add_argument('--replay_batch_newtask', help='重放批次中新任务样本数量', type=int, default=30)
    argparser.add_argument('--replay', help='是否重放', type=bool, default=True)
    # 可选参数为all、new、false
    argparser.add_argument('--recompute_proto', help='是否重计算原型', type=str, default='new')
    # 可选参数为all、base、false
    argparser.add_argument('--sdc', help='是否补偿语义漂移', type=str, default='base')
    argparser.add_argument('--random_test', help='是否随机抽取session数据', type=bool, default=True)
    argparser.add_argument('--r_xishu', help='正则化系数', type=float, default=0.05)
    argparser.add_argument('--test_times', help='测试次数', type=int, default=20)
    args = argparser.parse_args()
    all_result = cl_memory_triplet_train(args=args)
    print('重放批次:', args.replay_batch,'包含新任务样本',args.replay_batch_newtask, '个 重放次数:', args.replay_epoch, '微调次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：',
          args.r_xishu, ' 学习率：', args.cl_lr, ' 优化器：', args.cl_optim, ' sdc:', args.sdc, ' recompute:',
          args.recompute_proto)
    # print(all_result)
    print('avg result:')
    print(torch.Tensor(all_result).mean(dim=0))
    print(args)
    torch.cuda.empty_cache()
    print('**************************************************************************')
    print('**************************************************************************')
    # args.replay_epoch = 30
    # args.recompute_proto = 'new'
    # args.sdc = 'base'
    # all_result = cl_memory_triplet_train(args=args)
    # print('重放次数：', args.replay_epoch, '微调次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：',
    #       args.r_xishu, ' 学习率：', args.cl_lr, ' 优化器：', args.cl_optim, ' sdc:', args.sdc, ' recompute:',
    #       args.recompute_proto)
    # # print(all_result)
    # print('avg result:')
    # print(torch.Tensor(all_result).mean(dim=0))
    # print(args)
    # torch.cuda.empty_cache()
    # print('**************************************************************************')
    # print('**************************************************************************')
    # #######
    # args.session_epoch = 10
    # args.replay_epoch = 20
    # all_result = cl_memory_triplet_train(args=args)
    # print('avg result:')
    # print(torch.Tensor(all_result).mean(dim=0))
    # print(args)
    # print('**************************************************************************')
    # print('**************************************************************************')
    # print('**************************************************************************')
    # torch.cuda.empty_cache()
