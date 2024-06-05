# 对fewrel数据进行联合训练，分别使用fc和triplet loss进行训练
#
import torch
import torch.nn.functional as F
import numpy as np
from data_process.fewrel import Joint_dataset, FewRel_Bert
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
from models.Bert import Bert, Bert_fc, Bert_cnn, Bert_cnn_pos, Bert_entitystart
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import time


def train(model, device, train_loader, optimizer, epoch, loss_fn):
    return


def eval(model, device, val_loader, epoch, loss_fn):
    return


def triplet_train_and_eval(args):
    with open(args.bert_path + 'word2id.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)
    train_dataset = FewRel_Bert(args.data_path, max_length=args.max_length, mode='cl_train', bert_path=args.bert_path,
                                word2id=word2id)
    val_dataset = FewRel_Bert(args.data_path, max_length=args.max_length, mode='cl_test', bert_path=args.bert_path,
                              word2id=word2id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    prototype_loader = DataLoader(train_dataset, batch_size=140, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    model = Bert_entitystart(args)
    model.to(device)
    param_optimizer = list(model.named_parameters())  # 模型参数名字列表
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.lr,
    #                      warmup=0.05,
    #                      t_total=len(train_dataloader) * args.epochs
    #                      )
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epoch, gamma=0.1)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epoch, gamma=0.1)
    if args.losses == 'triplet':
        loss_fn = TripletLoss(args.margin)
    if args.losses == 'centertriplet':
        loss_fn = CenterTripletLoss()
    result_file = open(args.result_path, 'w')
    count = 0
    best_val = 0
    # 冻结bert除最后一层外参数 并查看可训练层
    for name, para in model.named_parameters():
        if '6' not in name:
            para.requires_grad = False
        else:
            break
    # 冻结bert参数 并查看可训练层
    # for name, para in model.named_parameters():
    #     if 'bert' not in name:
    #         para.requires_grad = True
    #     else:
    #         para.requires_grad = False
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        # 查看可训练层
        # for name, para in model.named_parameters():
        #     if para.requires_grad == True:
        #         print(name,end=' ')
        train_loss = 0
        for batch_idx, ((x1, x2, x3, pos1, pos2), y) in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            model.zero_grad()
            x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                device), y.to(device)
            logits = model([x1, x2, x3], pos1, pos2)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        print('epoch:{} train_loss:{:.4f}'.format(epoch, train_loss))
        # 测试阶段
        # 1 计算类原型
        model.eval()
        with torch.no_grad():
            prototypes = torch.Tensor(args.class_num, args.encoder_dim).to(device)
            temp_proto = torch.zeros(args.encoder_dim).to(device)
            for i, ((x1, x2, x3, pos1, pos2), y) in enumerate(prototype_loader):
                x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                    device), y.to(device)
                temp_proto += model([x1, x2, x3], pos1, pos2).sum(dim=0)
                if (i + 1) % 4 == 0:
                    prototypes[y[0].item()] = temp_proto / 560.0
                    temp_proto = torch.zeros(args.encoder_dim).to(device)
            # 2 计算训练精度
            correct = 0
            for (x1, x2, x3, pos1, pos2), y in train_dataloader:
                x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                    device), y.to(device)
                dist = euclidean_dist(model([x1, x2, x3], pos1, pos2), prototypes)
                _, pred = torch.max(-dist, -1)
                correct += (pred == y).sum().float()
            train_acc = correct / len(train_dataset)
            print('train_acc:{:.4f}'.format(train_acc))
            # 计算验证精度
            correct = 0
            for (x1, x2, x3, pos1, pos2), y in val_dataloader:
                x1, x2, x3, pos1, pos2, y = x1.to(device), x2.to(device), x3.to(device), pos1.to(device), pos2.to(
                    device), y.to(device)
                dist = euclidean_dist(model([x1, x2, x3], pos1, pos2), prototypes)
                _, pred = torch.max(-dist, -1)
                correct += (pred == y).sum().float()
            val_acc = correct / len(val_dataset)
        print('val_acc:{:.4f}'.format(val_acc))
        result_file.write(
            'epoch:{} tr_loss:{:.4f}  tr_acc:{:.4f}  val_acc:{:.4f} \n'.format(epoch, train_loss,
                                                                               train_acc,
                                                                               val_acc))
        # scheduler.step()
        if best_val < val_acc:
            best_val = val_acc
            best_epoch = epoch
            model.to(device_cpu)
            torch.save(model, args.model_save_path)
            model.to(device)
            count = 0
        elif count >= 30:
            break
        count += 1
    result_file.write('best_epoch:{}  best_val_acc:{:.4f}'.format(best_epoch, best_val))
    result_file.close()
    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--bert_path', type=str, default='D:/code/pretrained_model/bert/uncased_L-12_H-768_A-12/')
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='data/all_wiki.json')
    argparser.add_argument('--epochs', help='轮次', type=int,
                           default=200)
    argparser.add_argument('--max_length', help='文本最大长度', type=int,
                           default=38)
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=224)
    argparser.add_argument('--class_num', help='类别数量', type=int,
                           default=40)
    argparser.add_argument('--lr', help='学习率', type=float,
                           default=3e-4)
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--encoder_dim', type=int, default=1536)
    argparser.add_argument('--losses', help='loss function', type=str, default='triplet')
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='result/temp_starttag_bert.txt')  # base_bert_ft6_cnn_pos_triplet_adam3e-4_nd
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='result/temp_starttag_bert.pth')
    argparser.add_argument('--lr_decay_epoch', help='学习率衰减', type=list, default=[20, 40, 60, 80])
    argparser.add_argument('--weight_decay', help='', type=float, default=0.0)
    argparser.add_argument('--pos_embedding_dim', help='', type=int, default=5)
    argparser.add_argument('--optimizer', help='optimizer', type=str, default='adam')
    args = argparser.parse_args()
    print(args)
    triplet_train_and_eval(args)
    print(args)
