import torch
import torch.nn.functional as F
import numpy as np
from data_process.clinc150 import CLINC150_Glove, CLINC150_Bert
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from clinc_train.clinc_models.embedding import Glove_Embedding
from clinc_train.clinc_models.encoder import CNN_Encoder
from clinc_train.clinc_models.joint_model import Joint_model, Embed_model
import json
from losses.triplet import TripletLoss, TripletLoss_avg
from losses.CenterTriplet import CenterTripletLoss
from losses.ContrastiveLoss import ContrastiveLoss
from losses.metrics import euclidean_dist


def fc_train():
    return


def triplet_train(args):
    with open(args.word2id_path) as f:
        word2id = json.load(f)
    embedding = Glove_Embedding(word2id, embedding_dim=args.embedding_dim, root=args.pre_embedding_path,max_length=args.max_length)
    encoder = CNN_Encoder(max_length=args.max_length,word_embedding=args.embedding_dim, hidden_size=args.encoder_dim)
    model = Embed_model(embedding, encoder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    model.to(device)
    tr_dataset = CLINC150_Glove(args.data_path, word2id, max_length=args.max_length, mode='cl_train')
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    prototype_loader = DataLoader(tr_dataset, batch_size=100, shuffle=False)
    val_dataset = CLINC150_Glove(args.data_path, word2id, max_length=args.max_length, mode='cl_val')
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
        model.train()
        for sentence, label in tqdm(tr_dataloader):
            sentence = sentence.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            out_put = model(sentence)
            loss = loss_fn(out_put, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(tr_dataloader)
        print('*****epoch:{}*****'.format(epoch))
        print('tr_loss:{:.4f}'.format(train_loss))
        with torch.no_grad():
            # 计算类原型
            prototypes = torch.Tensor(args.class_num, args.encoder_dim).to(device)
            for sentence, label in prototype_loader:
                sentence = sentence.to(device)
                prototypes[label[0].item()] = model(sentence).mean(dim=0)
            # 计算训练精度
            correct = 0
            for sentence, label in tr_dataloader:
                sentence = sentence.to(device)
                label = label.to(device)
                dist = euclidean_dist(model(sentence), prototypes)
                _, pred = torch.max(-dist, -1)
                correct += (pred == label).sum().float()
            train_acc = correct / len(tr_dataset)
            print('train_acc:{:.4f}'.format(train_acc))
            # 计算验证精度
            correct = 0
            for sentence, label in val_dataloader:
                sentence = sentence.to(device)
                label = label.to(device)
                dist = euclidean_dist(model(sentence), prototypes)
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
        elif count >= 60:
            break
        count += 1
    result_file.write('best_epoch:{}  best_val_acc:{:.4f}'.format(best_epoch, best_val))
    result_file.close()
    return


def centertriplet_train():
    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='E:/ly/CLINC150/data_full.json')
    argparser.add_argument('--epoch', help='轮次', type=int,
                           default=300)
    argparser.add_argument('--class_num', help='类别数量', type=int,
                           default=70)
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=512)
    argparser.add_argument('--lr', help='学习率', type=float,
                           default=3e-4)
    argparser.add_argument('--word2id_path', type=str,
                           default='word2id.json')
    argparser.add_argument('--max_length', type=int, default=28)
    argparser.add_argument('--weight_decay', help='', type=float, default=1e-5)
    argparser.add_argument('--embedding_dim', type=int, default=100)
    argparser.add_argument('--encoder_dim', type=int, default=512)
    argparser.add_argument('--lr_decay_epoch', help='学习率衰减', type=list, default=[40, 100, 150, 180])
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--losses', help='loss function', type=str, default='triplet')
    argparser.add_argument('--optimizer', help='optimizer', type=str, default='adam')
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='clinc_result/clinc_base_glove_ftall_triplet_m0.5_decay_adam3e-4.txt')
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='clinc_result/clinc_base_glove_ftall_triplet_m0.5_decay_adam3e-4.pth')
    argparser.add_argument('--pre_embedding_path',  help='glove词嵌入位置',type=str, default='E:/ly/增量学习代码/glove.6B')
    args = argparser.parse_args()
    triplet_train(args)
