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
    # memory = []
    prototypes = torch.Tensor(args.class_num, args.encoder_dim).to(device)
    with torch.no_grad():
        temp_proto = torch.zeros(args.encoder_dim).to(device)
        for i, ((x1, x2, x3, pos1, pos2), y) in enumerate(prototype_loader):
            x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                device), y.to(device)
            temp_proto += model([x1, x2, x3]).sum(dim=0)
            if (i + 1) % 4 == 0:
                prototypes[y[0].item()] = temp_proto / 560.0
                temp_proto = torch.zeros(args.encoder_dim).to(device)
    fix_prototypes = copy.deepcopy(prototypes)
    all_result = []
    for times in range(args.test_times):
        start_time = time.time()
        session_loss = [0.0] * 9
        session_acc = [0.0] * 9
        session_acc[0] = 0.6821
        session_dataset = Bert_Session(args.data_path, word2id, 36, 1, 5, 'val')
        model = copy.deepcopy(fix_model)
        model = model.to(device)
        prototypes = copy.deepcopy(fix_prototypes)
        # 冻结bert除6层前参数 并查看可训练层
        for name, para in model.named_parameters():
            if '6' not in name:
                para.requires_grad = False
            else:
                break
        if args.cl_optim == 'sgd':
            cl_optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        elif args.cl_optim == 'adam':
            cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)

        print('times:', times, end='')
        # # 得到前40类样本随机每类抽取5个
        # for cls in range(0,40):
        #     for memory_idx in random.sample(range(0,560), args.K_shot):
        #         memory.append(train_dataset.__getitem__(cls*560+memory_idx))
        for i in range(1, 9):
            # 增量学习
            print('# ', end='')
            session_dataset.set_session(i)
            session_dataloader = DataLoader(session_dataset, batch_size=args.batch_size, shuffle=False)
            (x1, x2, x3, pos1, pos2), y = session_dataset.get_random_train_session()
            x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                device), y.to(device)
            if args.regularization == 'lwf':
                old_model = copy.deepcopy(model)
                old_model.eval()
                for paras in old_model.parameters():
                    paras.requires_grad = False
                old_out_put = old_model([x1, x2, x3])
            model.train()
            for j in range(args.session_epoch):
                cl_optimizer.zero_grad()
                logits = model([x1, x2, x3])
                loss = torch.tensor(0.0)
                if args.regularization == 'lwf':
                    loss = torch.norm(logits-old_out_put, dim=1).sum()
                ny = torch.clone(y)
                for memory_idx in random.sample(range(0, i * 5 + 35), args.replay_num):
                    logits = torch.cat([logits,prototypes[memory_idx].unsqueeze(0)],dim=0)
                    ny = torch.cat([ny, torch.tensor([memory_idx]).to(device)], dim=0)
                loss += loss_fn(logits, ny)
                loss.backward()
                cl_optimizer.step()
                session_loss[i] += loss.item()
            session_loss[i] = round(session_loss[i] / (args.session_epoch + 0.0001), 4)
            # 在所有遇见过的类上测试
            correct = 0
            with torch.no_grad():
                # 得到新的类的原型
                model.eval()
                out_put = model([x1, x2, x3])
                out_put = out_put.view(5, args.K_shot, args.encoder_dim)
                new_class_proto = out_put.mean(dim=1)
                prototypes[i * 5 + 35:i * 5 + 40] = new_class_proto
                for (x1, x2, x3, pos1, pos2), y in session_dataloader:
                    x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                        device), y.to(device)
                    dist = euclidean_dist(model([x1, x2, x3]), prototypes[:i * 5 + 40])
                    _, pred = torch.max(-dist, dim=-1)
                    correct += (pred == y).sum().float().item()
                session_acc[i] = round(correct / len(session_dataset), 4)
                # print('session:{} val_acc:{:.4f}'.format(i, session_acc[i]))
        print('\nacc:',session_acc)
        print('loss:',session_loss)
        print('以上结果超参配置：',args.session_epoch, ' ', args.regularization, ' ', args.cl_lr, ' ', args.cl_optim)
        all_result.append(session_acc)
        end_time = time.time()
        print('用时：', (end_time-start_time)/60,'分')
    return all_result

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bert_path', type=str, default='D:/code/pretrained_model/bert/uncased_L-12_H-768_A-12/')
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='data/all_wiki.json')
    argparser.add_argument('--epochs', help='轮次', type=int,
                           default=90)
    argparser.add_argument('--max_length', help='轮次', type=int,
                           default=36)
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=256)
    argparser.add_argument('--class_num', help='类别数量', type=int,
                           default=80)
    argparser.add_argument('--lr', help='学习率', type=float,
                           default=0.1)
    argparser.add_argument('--cl_lr', help='增量学习学习率', type=float,
                           default=1e-5)
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--encoder_dim', type=int, default=768)
    argparser.add_argument('--losses', help='loss function', type=str, default='triplet')
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='result/base_bert_ft6_cnn_pos_triplet.txt')
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='result/base_bert_ft6_cnn_pos_triplet.pth')
    argparser.add_argument('--weight_decay', help='', type=float, default=1e-5)
    argparser.add_argument('--test_times', help='测试次数', type=int, default=15)
    argparser.add_argument('--cl_optim',help='使用哪种优化器', type=str, default='adam')
    argparser.add_argument('--regularization',help='使用什么正则化 比如lwf',type=str,default='lwf')
    argparser.add_argument('--session_epoch', help='增量任务轮次', type=int, default=0)
    argparser.add_argument('--K_shot', help='每类样本数量', type=int, default=5)
    argparser.add_argument('--replay_num', help='使用多少个旧类原型进行知识重放', type=int,default=20)
    args = argparser.parse_args()
    all_result = cl_triplet_train(args=args)

    # result = torch.Tensor(args.test_times, 9)
    # for i in range(args.test_times):
    #     print('time:', i)
    #     acc, loss = cl_triplet_train(args=args)
    #     torch.cuda.empty_cache()
    #     result[i] = torch.Tensor(acc)
    print(all_result)
    print('avg result:')
    print(torch.Tensor(all_result).mean(dim=0))

    torch.cuda.empty_cache()
