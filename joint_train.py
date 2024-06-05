# 对fewrel数据进行联合训练，分别使用fc和triplet loss进行训练
#
import torch
import torch.nn.functional as F
import numpy as np
from data_process.fewrel import Joint_dataset
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


def fc_train(args):
    with open(args.word2id_path) as f:
        word2id = json.load(f)
    embedding = Glove_Embedding(word2id, embedding_dim=args.embedding_dim, root=args.pre_embedding_path)
    encoder = CNN_Encoder(word_embedding=args.embedding_dim, hidden_size=args.encoder_dim)
    model = Joint_model(embedding, encoder, 80)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    model.to(device)
    tr_dataset = Joint_dataset(args.data_path, word2id, max_length=36, mode='train')
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = Joint_dataset(args.data_path, word2id, max_length=36, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epoch, gamma=0.1)
    result_file = open(args.result_path, 'w')
    best_val = 0
    for epoch in range(args.epoch):
        train_loss = 0
        val_loss = 0
        model.train()
        correct = 0
        for (sentence, pos1, pos2, mask), label in tqdm(tr_dataloader):
            sentence = sentence.to(device)
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out_put = model(sentence, pos1, pos2)
            loss = F.cross_entropy(out_put, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            predict = torch.argmax(out_put, 1)
            correct += (predict == label).sum().float()
        train_acc = correct / len(tr_dataset)
        train_loss = train_loss / len(tr_dataloader)
        print('*****epoch:{}*****'.format(epoch))
        print('tr_loss:{:.4f}  tr_acc:{:.4f}'.format(train_loss, train_acc))
        model.eval()
        correct = 0
        with torch.no_grad():
            for (sentence, pos1, pos2, mask), label in tqdm(val_dataloader):
                sentence = sentence.to(device)
                pos1 = pos1.to(device)
                pos2 = pos2.to(device)
                label = label.to(device)
                out_put = model(sentence, pos1, pos2)
                loss = F.cross_entropy(out_put, label)
                val_loss += loss.item()
                predict = torch.argmax(out_put, 1)
                correct += (predict == label).sum().float()
        val_acc = correct / len(val_dataset)
        val_loss = val_loss / len(val_dataloader)
        print('val_loss:{:.4f}  val_acc:{:.4f}'.format(val_loss, val_acc))
        result_file.write(
            'epoch:{} tr_loss:{:.4f}  tr_acc:{:.4f} val_loss:{:.4f}  val_acc:{:.4f} \n'.format(epoch, train_loss,
                                                                                               train_acc, val_loss,
                                                                                               val_acc))
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
        scheduler.step()
    result_file.write('best_epoch:{} best_val_acc:{:.4f} '.format(best_epoch, best_val))
    result_file.close()
    return


def base_train(args):
    with open(args.word2id_path) as f:
        word2id = json.load(f)
    embedding = Glove_Embedding(word2id, embedding_dim=args.embedding_dim, root=args.pre_embedding_path)
    encoder = CNN_Encoder(word_embedding=args.embedding_dim, hidden_size=args.encoder_dim)
    model = Embed_model(embedding, encoder, 80)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    model.to(device)
    tr_dataset = Joint_dataset(args.data_path, word2id, max_length=36, mode='cl_train')
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    prototype_loader = DataLoader(tr_dataset, batch_size=560, shuffle=False)
    val_dataset = Joint_dataset(args.data_path, word2id, max_length=36, mode='cl_test')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epoch, gamma=0.1)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.losses == 'triplet':
        loss_fn = TripletLoss(args.margin)
    if args.losses == 'centertriplet':
        loss_fn = CenterTripletLoss()
    result_file = open(args.result_path, 'w')
    best_val = 0
    count = 0
    for epoch in range(args.epoch):
        train_loss = 0
        val_loss = 0
        model.train()
        correct = 0
        for (sentence, pos1, pos2, mask), label in tqdm(tr_dataloader):
            sentence = sentence.to(device)
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out_put = model(sentence, pos1, pos2)
            loss = loss_fn(out_put, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(tr_dataloader)
        print('*****epoch:{}*****'.format(epoch))
        print('tr_loss:{:.4f}'.format(train_loss))
        with torch.no_grad():
            # 计算类原型,首先得到所有embedding (如果是训练增量学习特征提取器，第一维度40，joint_train为80)
            prototypes = torch.Tensor(40, args.encoder_dim).to(device)
            for (sentence, pos1, pos2, mask), label in prototype_loader:
                sentence = sentence.to(device)
                pos1 = pos1.to(device)
                pos2 = pos2.to(device)
                prototypes[label[0].item()] = model(sentence, pos1, pos2).mean(dim=0)
            # 计算训练精度
            correct = 0
            for (sentence, pos1, pos2, mask), label in tr_dataloader:
                sentence = sentence.to(device)
                pos1 = pos1.to(device)
                pos2 = pos2.to(device)
                label = label.to(device)
                dist = euclidean_dist(model(sentence, pos1, pos2), prototypes)
                _, pred = torch.max(-dist, -1)
                correct += (pred == label).sum().float()
            train_acc = correct / len(tr_dataset)
            print('train_acc:{:.4f}'.format(train_acc))
            # 计算验证精度
            correct = 0
            for (sentence, pos1, pos2, mask), label in val_dataloader:
                sentence = sentence.to(device)
                pos1 = pos1.to(device)
                pos2 = pos2.to(device)
                label = label.to(device)
                dist = euclidean_dist(model(sentence, pos1, pos2), prototypes)
                _, pred = torch.max(-dist, -1)
                correct += (pred == label).sum().float()
            val_acc = correct / len(val_dataset)
            print('val_acc:{:.4f}'.format(val_acc))
        if args.optimizer == 'sgd':
            scheduler.step()
        result_file.write(
            'epoch:{} tr_loss:{:.4f}  tr_acc:{:.4f}  val_acc:{:.4f} \n'.format(epoch, train_loss,
                                                                               train_acc,
                                                                               val_acc))
        if best_val < val_acc:
            best_val = val_acc
            best_epoch = epoch
            model.to(device_cpu)
            torch.save(model, args.model_save_path)
            model.to(device)
            count = 0
        elif count >= 50:
            break
        count += 1

    result_file.write('best_epoch:{}  best_val_acc:{:.4f}'.format(best_epoch, best_val))
    result_file.close()
    return


def compute_prototype(model, data):
    return


def triplet_test(args, model, dataset):
    return


if __name__ == '__main__':
    print()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='data/all_wiki.json')
    argparser.add_argument('--epoch', help='轮次', type=int,
                           default=300)
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=256)
    argparser.add_argument('--lr', help='学习率', type=float,
                           default=1e-3)
    argparser.add_argument('--word2id_path', type=str,
                           default='data/word2id.json')
    argparser.add_argument('--weight_decay', help='', type=float, default=1e-5)
    argparser.add_argument('--embedding_dim', type=int, default=100)
    argparser.add_argument('--encoder_dim', type=int, default=512)
    argparser.add_argument('--lr_decay_epoch', help='学习率衰减', type=list, default=[20, 40, 60, 80])
    argparser.add_argument('--optimizer', help='优化器', type=str, default='adam')
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--losses', help='loss function', type=str, default='triplet')
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='result/base_glove_triplet_m0.5_adam1e-3_encoder512_pos_decay.txt')
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='result/base_glove_triplet_m0.5_adam1e-3_encoder512_nopos_decay.pth')
    argparser.add_argument('--pre_embedding_path', help='glove词嵌入位置', type=str, default='E:/ly/增量学习代码/glove.6B')
    args = argparser.parse_args()

    base_train(args)
    # 以margin为1进行训练
    args.margin = 1
    args.result_path = 'result/base_glove_triplet_m1_adam1e-3_encoder512_pos_decay.txt'
    args.model_save_path = 'result/base_glove_triplet_m1_adam1e-3_encoder512_pos_decay.pth'
    base_train(args)
    # 以margin为1  不带权重衰减进行训练
    args.margin = 1
    args.weight_decay = 0.0
    args.result_path = 'result/base_glove_triplet_m1_adam1e-3_encoder512_pos_nodecay.txt'
    args.model_save_path = 'result/base_glove_triplet_m1_adam1e-3_encoder512_pos_nodecay.pth'
    base_train(args)
    # 以margin为0.5 不带权重衰减进行训练
    args.margin = 0.5
    #四个中下面这个精度最高
    args.result_path = 'result/base_glove_triplet_m0.5_adam1e-3_encoder512_pos_nodecay.txt'
    args.model_save_path = 'result/base_glove_triplet_m0.5_adam1e-3_encoder512_pos_nodecay.pth'
    base_train(args)
