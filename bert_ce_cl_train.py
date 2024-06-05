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
from models.Bert import Bert,Bert_fc,Bert_cnn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import time
import copy

def cl_ce_train(args):
    with open(args.bert_path + 'word2id.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    train_dataset = FewRel_Bert(args.data_path, max_length=36, mode='cl_train', bert_path=args.bert_path,
                                word2id=word2id)
    # 用于生成每个session的训练数据和测试数据
    session_dataset = Bert_Session(args.data_path, word2id, 36, 1, 5, 'val')
    prototype_loader = DataLoader(train_dataset, batch_size=140, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    #需要考虑模型参数不同问题
    model40 = torch.load(args.model_save_path)
    model = Bert_fc(args)
    # ？？？？！！！以此方法复制参数会不会导致无法正常反向传播需要考虑
    model.fc.weight[:40] = copy.deepcopy(model40.fc.weight)
    fix_model = copy.deepcopy(model)
    model.to(device)
    if args.cl_optim == 'sgd':
        cl_optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
    elif args.cl_optim == 'adam':
        cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    result_file = open(args.result_path, 'w')
    # 设置6层后参数可训练
    for name, para in model.named_parameters():
        if '6' not in name:
            para.requires_grad = False
        else:
            break
    for name, para in model.named_parameters():
        para.requires_grad = True
    all_result = torch.Tensor(args.test_times, 9)
    for times in range(args.test_times):
        start_time = time.time()
        session_loss = [0.0] * 9
        session_acc = [0.0] * 9
        session_acc[0] = 0.6821
        session_dataset = Bert_Session(args.data_path, word2id, 36, 1, 5, 'val')
        model = copy.deepcopy(fix_model)
        model = model.to(device)
        print('times:',times,end='')
        for i in range(1,9):
            print('# ',end='')
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
                logits = model([x1,x2,x3])
                loss = loss_fn(logits,y)
                if args.regularization == 'lwf':
                    loss += torch.norm(logits - old_out_put, dim=1).sum()
                loss.backward()
                cl_optimizer.step()
                session_loss[i] += loss.item()
            session_loss[i] = round(session_loss[i] / (args.session_epoch + 0.0001), 4)
            #再所有遇见过的类上测试：
            with torch.no_grad():
                model.eval()
                acc = 0.0
                for (x1, x2, x3, pos1, pos2), y in session_dataloader:
                    x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                        device), y.to(device)
                    logits = model([x1, x2, x3])
                    pred = logits.max(-1, keepdim=True)[1]
                    acc += pred.eq(y.view_as(pred)).sum().item()
                acc /=len(session_dataloader.dataset)
                session_acc[i] = round(acc, 4)
        print('acc:', session_acc)
        print('loss:', session_loss)
        print('以上结果超参配置：', args.session_epoch, ' ', args.regularization, ' ', args.cl_lr, ' ', args.cl_optim)
        all_result[i] = torch.Tensor(session_acc)
        end_time = time.time()
        print('用时：', (end_time - start_time) / 60, '分')
    return all_result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bert_path', type=str, default='E:/数据集and预训练模型/bert/uncased_L-12_H-768_A-12/')
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='data/all_wiki.json')
    argparser.add_argument('--epochs', help='轮次', type=int,
                           default=90)
    argparser.add_argument('--max_length', help='轮次', type=int,
                           default=36)
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=4)
    argparser.add_argument('--class_num', help='类别数量', type=int,
                           default=80)
    argparser.add_argument('--lr', help='学习率', type=float,
                           default=0.1)
    argparser.add_argument('--cl_lr', help='增量学习学习率', type=float,
                           default=1e-4)
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--encoder_dim', type=int, default=768)
    argparser.add_argument('--losses', help='loss function', type=str, default='triplet')
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='result/base_bert_ft6_fc_ce_nopos.txt')
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='result/base_bert_ft6_fc_ce_nopos.pth')
    argparser.add_argument('--lr_decay_epoch', help='学习率衰减', type=list, default=[15, 25, 35, 45])
    argparser.add_argument('--weight_decay', help='', type=float, default=1e-5)
    argparser.add_argument('--reuse', help='是否使用训练过的模型', type=bool, default=True)
    argparser.add_argument('--test_times', help='测试次数', type=int, default=5)
    argparser.add_argument('--cl_optim',help='使用哪种优化器', type=str, default='adam')
    argparser.add_argument('--regularization',help='使用什么正则化 比如lwf',type=str,default='lwf')
    argparser.add_argument('--session_epoch', help='增量任务轮次', type=int, default=10)
    argparser.add_argument('--K_shot', help='每类样本数量', type=int, default=5)
    args = argparser.parse_args()
    # all_result = cl_ce_train(args=args)
    # print('avg result:')
    # print(all_result.mean(dim=0))
    # torch.cuda.empty_cache()
