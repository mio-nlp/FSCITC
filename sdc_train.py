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


def cl_train(args):
    with open(args.word2id_path) as f:
        word2id = json.load(f)
    embedding = Glove_Embedding(word2id, embedding_dim=args.embedding_dim)
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
    result_file = open(args.result_path, 'w')
    best_val = 0
    # 定义类原型
    prototypes = torch.zeros(80, args.encoder_dim).float().to(device)
    if not args.reuse_base:  # 在session[0]上训练
        return
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
    session_data = Session(args.data_path, word2id, 36, session=1, K=args.K_shot)
    # 共8个session
    session_loss = [0.0] * 9
    session_acc = [0.0] * 9
    session_acc[0] = 0.8159

    for i in range(1, 9):
        # train
        session_data.set_session(i)
        session_dataloader = DataLoader(session_data, batch_size=args.batch_size, shuffle=False)
        (sentence, pos1, pos2, mask), label = session_data.get_random_train_session()
        sentence = sentence.to(device)
        pos1 = pos1.to(device)
        pos2 = pos2.to(device)
        label = label.to(device)
        old_model = copy.deepcopy(model)
        old_model.eval()
        for paras in old_model.parameters():
            paras.requires_grad = False
        old_out_put = old_model(sentence, pos1, pos2)
        model.train()
        for epoch in range(args.session_epoch):
            cl_optimizer.zero_grad()
            out_put = model(sentence, pos1, pos2)
            loss = loss_fn(out_put, label)
            if args.regularization == 'lwf':
                loss += torch.norm(out_put - old_out_put, dim=1).sum()
            loss.backward()
            session_loss[i] += loss.item()
            cl_optimizer.step()
        if args.sdc:
            weight = compute_sample_weight(prototypes, out_put)  # weight: [80,n*k]
            for cls_index in range(35 + i * 5):
                compensation = ((out_put - old_out_put) * weight[cls_index].reshape(5 * args.K_shot, 1) / weight[
                    cls_index].sum()).sum(dim=0)
                prototypes[cls_index] = prototypes[cls_index] + compensation

        session_loss[i] = round(session_loss[i] / (args.session_epoch + 0.0001), 4)
        # 在所有遇见过的类上测试
        correct = 0
        with torch.no_grad():
            # 得到新的类的原型
            model.eval()
            out_put = out_put.view(5, args.K_shot, args.encoder_dim)
            new_class_proto = out_put.mean(dim=1)
            prototypes[i * 5 + 35:i * 5 + 40] = new_class_proto
            # 在所有遇见过的类上测试
            # print(len(session_data), ' ', len(session_dataloader))
            for (sentence, pos1, pos2, mask), label in session_dataloader:
                sentence = sentence.to(device)
                pos1 = pos1.to(device)
                pos2 = pos2.to(device)
                label = label.to(device)
                dist = euclidean_dist(model(sentence, pos1, pos2), prototypes[:i * 5 + 40])
                _, pred = torch.max(-dist, dim=-1)
                correct += (pred == label).sum().float().item()
            session_acc[i] = round(correct / len(session_data), 4)
            # print('session:{} val_acc:{:.4f}'.format(i, session_acc[i]))
    print(session_acc)
    print(session_loss)
    print(args.session_epoch, ' ', args.regularization, ' ', args.cl_lr, ' ', args.cl_optim)
    return session_acc, session_loss


def base_train(args, ):
    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='E:/数据集and预训练模型/FewRel/FewRel1.0/all_wiki.json')
    argparser.add_argument('--base_epoch', help='轮次', type=int, default=100)
    argparser.add_argument('--session_epoch', help='增量任务轮次', type=int, default=100)
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=128)
    argparser.add_argument('--lr', help='学习率', type=float,
                           default=0.1)
    argparser.add_argument('--cl_lr', help='增量学习学习率', type=float,
                           default=1e-3)
    argparser.add_argument('--word2id_path', type=str,
                           default='E:/pythoncode/pytorch/myfewshotcode/fewrel再版/data/FewRel1.0/word2id.json')
    argparser.add_argument('--weight_decay', help='', type=float, default=0)
    argparser.add_argument('--embedding_dim', type=int, default=100)
    argparser.add_argument('--encoder_dim', type=int, default=230)
    argparser.add_argument('--base_lr_decay_epoch', help='学习率衰减', type=list, default=[20, 40, 60, 80])
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--K_shot', help='每类样本数量', type=int, default=5)
    argparser.add_argument('--reuse_base', help='是否使用预训练好的度量网络', type=bool, default=True)
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='E:/pythoncode/pytorch/myfewshotcode/CIFSLNLP/result/joint_triplet_cl_train.txt')
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='E:/pythoncode/pytorch/myfewshotcode/CIFSLNLP/result/triplet_base_m5_decay.pth')
    argparser.add_argument('--regularization', help='使用什么正则化 比如lwf', type=str, default='lwf')
    argparser.add_argument('--cl_optim', help='使用哪种优化器', type=str, default='adam')
    argparser.add_argument('--sdc', help='是否使用特征补偿', type=bool, default=False)
    argparser.add_argument('--test_times', help='测试次数', type=int, default=50)
    args = argparser.parse_args()
    result = torch.Tensor(args.test_times,9)
    for i in range(args.test_times):
        print('time:',i)
        acc, loss = cl_train(args=args)
        result[i] = torch.Tensor(acc)
    print('avg result:')
    print(result.mean(dim=0))
    # torch.cuda.empty_cache()
