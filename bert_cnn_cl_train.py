import torch
import torch.nn.functional as F
import numpy as np
from data_process.fewrel import Joint_dataset, FewRel_Bert, Bert_Session
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.embedding import Glove_Embedding
from models.sentence_encoder import CNN_Encoder
from models.joint_model import Joint_model, Embed_model
import json
from losses.triplet import TripletLoss, TripletLoss_avg
from losses.CenterTriplet import CenterTripletLoss
from losses.ContrastiveLoss import ContrastiveLoss
from losses.metrics import euclidean_dist
from models.Bert import Bert
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import time
import copy
import random

def cl_triplet_train(args):
    random.seed(0)
    with open(args.bert_path + 'word2id.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    train_dataset = FewRel_Bert(args.data_path, max_length=36, mode='cl_train', bert_path=args.bert_path,
                                word2id=word2id)
    # 用于生成每个session的训练数据和测试数据
    prototype_loader = DataLoader(train_dataset, batch_size=140, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    model = torch.load(args.model_save_path)
    fix_model = copy.deepcopy(model)
    model.to(device)
    loss_fn = TripletLoss(args.margin)
    # 计算原型
    model.eval()
    prototypes = torch.Tensor(args.class_num, args.encoder_dim).to(device)
    with torch.no_grad():
        temp_proto = torch.zeros(args.encoder_dim).to(device)
        for i, ((x1, x2, x3, pos1, pos2), y) in enumerate(prototype_loader):
            x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                device), y.to(device)
            temp_proto += model([x1, x2, x3], pos1, pos2).sum(dim=0)
            if (i + 1) % 4 == 0:
                prototypes[y[0].item()] = temp_proto / 560.0
                temp_proto = torch.zeros(args.encoder_dim).to(device)
    fix_prototypes = copy.deepcopy(prototypes)
    all_result = []
    # 计算base session精度
    session_dataset = Bert_Session(args.data_path, word2id, 36, 0, args.K_shot, 'cl_test')
    session_dataloader = DataLoader(session_dataset, batch_size=args.batch_size, shuffle=False)
    correct = 0
    for batch_idx, ((x1, x2, x3, pos1, pos2), y) in enumerate(session_dataloader):
        x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
            device), y.to(device)
        dist = euclidean_dist(model([x1, x2, x3], pos1, pos2), prototypes[:40])
        _, pred = torch.max(-dist, dim=-1)
        correct += (pred == y).sum().float().item()
    first_acc = correct / len(session_dataset)

    for times in range(args.test_times):
        start_time = time.time()
        session_loss = [0.0] * 9
        session_acc = [0.0] * 9
        session_acc[0] = first_acc
        session_dataset = Bert_Session(args.data_path, word2id, 36, 0, args.K_shot, 'cl_test')
        model = copy.deepcopy(fix_model)
        model = model.to(device)
        if args.cl_optim == 'sgd':
            cl_optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        elif args.cl_optim == 'adam':
            cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        prototypes = copy.deepcopy(fix_prototypes)
        # 冻结bert参数 并查看可训练层
        for para in model.parameters():
            para.requires_grad = True
        for name, para in model.named_parameters():
            if '6' in name:
                para.requires_grad = False
            else:
                break
        print('times:', times)
        for i in range(1, 9):
            every_session_acc = [0.0] * 9
            # 增量学习
            print(i, ' # ', end='')
            session_dataset.set_session(i)
            session_dataloader = DataLoader(session_dataset, batch_size=args.batch_size, shuffle=False)
            if args.random_test:
                (x1, x2, x3, pos1, pos2), y = session_dataset.get_random_train_session()
            else:
                (x1, x2, x3, pos1, pos2), y = session_dataset.get_train_session()
            x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                device), y.to(device)
            if args.regularization == 'lwf':
                old_model = copy.deepcopy(model)
                old_model.eval()
                for paras in old_model.parameters():
                    paras.requires_grad = False
                old_out_put = old_model([x1, x2, x3], pos1, pos2)
            if args.regularization == 'nl2':
                old_model = copy.deepcopy(model)
                old_model.eval()
                for paras in old_model.parameters():
                    paras.requires_grad = False
            model.train()
            for j in range(args.session_epoch):
                cl_optimizer.zero_grad()
                logits = model([x1, x2, x3], pos1, pos2)
                loss = loss_fn(logits, y)
                if args.regularization == 'lwf':
                    loss += (torch.norm(logits - old_out_put, dim=1).sum()) * args.r_xishu
                if args.regularization == 'nl2':
                    for key in model.state_dict().keys():
                        loss += args.r_xishu * (
                            torch.pow(model.state_dict()[key] - old_model.state_dict()[key], exponent=2).sum())
                loss.backward()
                cl_optimizer.step()
                session_loss[i] += loss.item()
            session_loss[i] = round(session_loss[i] / (args.session_epoch + 0.0001), 4)
            # 在所有遇见过的类上测试
            correct = 0
            with torch.no_grad():
                # 得到新的类的原型
                model.eval()
                out_put = model([x1, x2, x3], pos1, pos2)
                out_put = out_put.view(5, args.K_shot, args.encoder_dim)
                new_class_proto = out_put.mean(dim=1)
                prototypes[i * 5 + 35:i * 5 + 40] = new_class_proto
                for batch_idx, ((x1, x2, x3, pos1, pos2), y) in enumerate(session_dataloader):
                    x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                        device), y.to(device)
                    dist = euclidean_dist(model([x1, x2, x3], pos1, pos2), prototypes[:i * 5 + 40])
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
                session_acc[i] = round(correct / len(session_dataset), 4)
                every_session_acc[0] = round(every_session_acc[0] / 5600, 3)
                for idx in range(1, 9):
                    every_session_acc[idx] = round(every_session_acc[idx] / 700, 3)
                # print(every_session_acc)
                # print('session:{} val_acc:{:.4f}'.format(i, session_acc[i]))
        print('\nacc:', session_acc)
        print('loss:', session_loss)
        all_result.append(session_acc)
        end_time = time.time()
        print('用时：', (end_time - start_time) / 60, '分')
    return all_result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bert_path', type=str, default='E:/ly/bert/uncased_L-12_H-768_A-12/')
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='data/all_wiki.json')
    argparser.add_argument('--max_length', help='轮次', type=int,
                           default=36)
    argparser.add_argument('--batch_size', type=int, default=80)
    argparser.add_argument('--class_num', help='类别数量', type=int,
                           default=80)

    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--encoder_dim', type=int, default=512)
    argparser.add_argument('--losses', help='loss function', type=str, default='triplet')
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='result/base_bert_ft6_cnn_pos_triplet.pth') #base_bert_ft6_cnn_pos_triplet
    argparser.add_argument('--K_shot', help='每类样本数量', type=int, default=10)
    argparser.add_argument('--weight_decay', help='', type=float, default=1e-5)
    argparser.add_argument('--test_times', help='测试次数', type=int, default=1)
    argparser.add_argument('--cl_optim', help='使用哪种优化器', type=str, default='adam')
    argparser.add_argument('--regularization', help='使用什么正则化 比如lwf', type=str, default='lwf')
    argparser.add_argument('--session_epoch', help='增量任务轮次', type=int, default=0)
    argparser.add_argument('--cl_lr', help='增量学习学习率', type=float,
                           default=1e-5)
    argparser.add_argument('--r_xishu', help='正则化系数', type=float, default=10)
    argparser.add_argument('--random_test', help='是否随机抽取session数据', type=bool, default=True)
    args = argparser.parse_args()
    all_result = cl_triplet_train(args=args)
    print('梯度下降次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：', args.r_xishu, ' 学习率：', args.cl_lr,
          ' 优化器：', args.cl_optim, 'K_shot:', args.K_shot)
    if args.test_times > 1:
        print(args.test_times, '次avg result:')
        print(torch.Tensor(all_result).mean(dim=0))
    torch.cuda.empty_cache()
