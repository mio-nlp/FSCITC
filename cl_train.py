import torch
import torch.nn.functional as F
import numpy as np
from data_process.fewrel import Joint_dataset, Session
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.embedding import Glove_Embedding
from models.sentence_encoder import CNN_Encoder
from models.joint_model import Joint_model, Embed_model
import json
from losses.triplet import TripletLoss
from losses.metrics import euclidean_dist
import copy
import random


def cl_train(args):
    random.seed(0)
    with open(args.word2id_path) as f:
        word2id = json.load(f)
    embedding = Glove_Embedding(word2id, embedding_dim=args.embedding_dim, root=args.glove_path)
    encoder = CNN_Encoder(word_embedding=args.embedding_dim, hidden_size=args.encoder_dim)
    model = Embed_model(embedding, encoder, 80)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    model.to(device)
    tr_dataset = Joint_dataset(args.data_path, word2id, max_length=36, mode='cl_train')
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    prototype_loader = DataLoader(tr_dataset, batch_size=560, shuffle=False)
    val_dataset = Joint_dataset(args.data_path, word2id, max_length=36, mode='cl_val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.base_lr_decay_epoch, gamma=0.1)
    loss_fn = TripletLoss(args.margin)
    best_val = 0
    # 定义类原型
    prototypes = torch.zeros(80, args.encoder_dim).float().to(device)
    if not args.reuse_base:  # 在session[0]上训练
        result_file = open(args.result_path, 'w')
        for epoch in range(args.base_epoch):
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
                # 计算类原型,首先得到所有embedding
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
                    dist = euclidean_dist(model(sentence, pos1, pos2), prototypes[:40])
                    _, pred = torch.max(-dist, -1)
                    correct += (pred == label).sum().float()
                val_acc = correct / len(val_dataset)
                print('session:0  val_acc:{:.4f}'.format(val_acc))
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
        result_file.write('best_epoch:{}  best_val_acc:{:.4f}'.format(best_epoch, best_val))
        result_file.close()
    else:  # 重用训练好的模型
        model = torch.load(args.model_save_path)
        model = model.to(device)
        # 计算原型
        for (sentence, pos1, pos2, mask), label in prototype_loader:
            sentence = sentence.to(device)
            pos1 = pos1.to(device)
            pos2 = pos2.to(device)
            prototypes[label[0].item()] = model(sentence, pos1, pos2).mean(dim=0)
        # 计算session 0 验证精度
        # correct = 0
        # for (sentence, pos1, pos2, mask), label in val_dataloader:
        #     sentence = sentence.to(device)
        #     pos1 = pos1.to(device)
        #     pos2 = pos2.to(device)
        #     label = label.to(device)
        #     dist = euclidean_dist(model(sentence, pos1, pos2), prototypes[:40])
        #     _, pred = torch.max(-dist, -1)
        #     correct += (pred == label).sum().float()
        # val_acc = correct / len(val_dataset)
        # # print('session:0  val_acc:{:.4f}'.format(val_acc))
        # best_val = val_acc
    # session0训练结束
    if args.cl_optim == 'sgd':
        cl_optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
    elif args.cl_optim == 'adam':
        cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
    session_data = Session(args.data_path, word2id, 36, session=1, K=args.K_shot,mode='cl_test')
    # print(session_data.random_data_index)
    # 共8个session
    session_loss = [0.0] * 9
    session_acc = [0.0] * 9
    session_acc[0] = 0.8044

    for i in range(1, 9):
        print('#', i, end='')
        every_session_acc = [0.0] * 9
        # train
        session_data.set_session(i)
        session_dataloader = DataLoader(session_data, batch_size=args.batch_size, shuffle=False)
        if args.random_test:
            (sentence, pos1, pos2, mask), label = session_data.get_random_train_session()
        else:
            (sentence, pos1, pos2, mask), label = session_data.get_train_session()
        sentence = sentence.to(device)
        pos1 = pos1.to(device)
        pos2 = pos2.to(device)
        label = label.to(device)
        if args.regularization == 'lwf':
            old_model = copy.deepcopy(model)
            old_model.eval()
            for paras in old_model.parameters():
                paras.requires_grad = False
            old_out_put = old_model(sentence, pos1, pos2)
        if args.regularization == 'nl2':
            old_model = copy.deepcopy(model)
            old_model.eval()
            for paras in old_model.parameters():
                paras.requires_grad = False
        model.train()
        for epoch in range(args.session_epoch):
            cl_optimizer.zero_grad()
            out_put = model(sentence, pos1, pos2)
            loss = loss_fn(out_put, label)
            if args.regularization == 'lwf':
                loss += args.r_xishu * (torch.norm(out_put - old_out_put, dim=1).sum())
            if args.regularization == 'nl2':
                for key in model.state_dict().keys():
                    loss += args.r_xishu * (
                        torch.pow(model.state_dict()[key] - old_model.state_dict()[key], exponent=2).sum())
            loss.backward()
            session_loss[i] += loss.item()
            cl_optimizer.step()
        session_loss[i] = round(session_loss[i] / (args.session_epoch + 0.0001), 4)
        # 在所有遇见过的类上测试
        correct = 0
        with torch.no_grad():
            # 得到新的类的原型
            model.eval()
            out_put = model(sentence, pos1, pos2)
            out_put = out_put.view(5, args.K_shot, args.encoder_dim)
            new_class_proto = out_put.mean(dim=1)
            prototypes[i * 5 + 35:i * 5 + 40] = new_class_proto
            # 在所有遇见过的类上测试
            # print(len(session_data), ' ', len(session_dataloader))
            for batch_idx, ((sentence, pos1, pos2, mask), label) in enumerate(session_dataloader):
                sentence = sentence.to(device)
                pos1 = pos1.to(device)
                pos2 = pos2.to(device)
                label = label.to(device)
                dist = euclidean_dist(model(sentence, pos1, pos2), prototypes[:i * 5 + 40])
                _, pred = torch.max(-dist, dim=-1)
                correct += (pred == label).sum().float().item()
                correct_num = (pred == label).sum().float().item()
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
            # print(every_session_acc)
            # print('session:{} val_acc:{:.4f}'.format(i, session_acc[i]))
    print('\nacc:', session_acc)
    print('loss:', session_loss)
    # print('梯度下降次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：', args.r_xishu, ' 学习率：', args.cl_lr,
    #       ' 优化器：', args.cl_optim, 'K_shot:', args.K_shot)
    return session_acc, session_loss


def base_train(args, ):
    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='data/all_wiki.json')
    argparser.add_argument('--base_epoch', help='轮次', type=int, default=100)
    argparser.add_argument('--glove_path', type=str, default='E:/ly/增量学习代码/glove.6B')
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=80)
    argparser.add_argument('--lr', help='学习率', type=float,
                           default=0.1)
    argparser.add_argument('--cl_lr', help='增量学习学习率', type=float,
                           default=5e-4)
    argparser.add_argument('--word2id_path', type=str,
                           default='data/word2id.json')
    argparser.add_argument('--weight_decay', help='', type=float, default=0)
    argparser.add_argument('--embedding_dim', type=int, default=100)
    argparser.add_argument('--encoder_dim', type=int, default=230)
    argparser.add_argument('--base_lr_decay_epoch', help='学习率衰减', type=list, default=[20, 40, 60, 80])
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--K_shot', help='每类样本数量', type=int, default=10)
    argparser.add_argument('--reuse_base', help='是否使用预训练好的度量网络', type=bool, default=True)
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='result/base_glove_triplet_m0.5_adam1e-3_encoder230_pos_decay.txt')
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='result/base_glove_triplet_m0.5_adam1e-3_encoder230_pos_decay.pth')
    argparser.add_argument('--regularization', help='使用什么正则化 比如lwf', type=str, default='no')
    argparser.add_argument('--r_xishu', help='正则化系数', type=float, default=1)
    argparser.add_argument('--session_epoch', help='增量任务轮次', type=int, default=10)
    argparser.add_argument('--cl_optim', help='使用哪种优化器', type=str, default='adam')
    argparser.add_argument('--test_times', help='测试次数', type=int, default=50)
    argparser.add_argument('--random_test', help='是否随机抽取session数据', type=bool, default=True)
    args = argparser.parse_args()
    # cl_train(args)
    result = torch.Tensor(args.test_times, 9)
    for i in range(args.test_times):
        print('time:', i)
        acc, loss = cl_train(args=args)
        result[i] = torch.Tensor(acc)
    print('avg result:')
    print(result.mean(dim=0))
    print('梯度下降次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：', args.r_xishu, ' 学习率：', args.cl_lr,
          ' 优化器：', args.cl_optim, 'K_shot:', args.K_shot)
    torch.cuda.empty_cache()
