import torch
import torch.nn.functional as F
import numpy as np
from data_process.clinc150 import Glove_session, Bert_session, CLINC150_Glove, CLINC150_Bert
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from clinc_train.clinc_models.embedding import Glove_Embedding
from clinc_train.clinc_models.encoder import CNN_Encoder
from clinc_train.clinc_models.joint_model import Joint_model, Embed_model
import json
from losses.CenterTriplet import CenterTripletLoss
from losses.triplet import TripletLoss
from losses.metrics import euclidean_dist
import copy
import time
import random


def compute_sample_weight(prototypes, sample_embeddings):
    # prototypes:[class_num, embed_size]
    # sample_embed:[n*k, embed_size]
    # weight:[class_num, n*k]  只有当前训练集见过的类别有意义
    class_num = prototypes.size()[0]
    embedd_size = prototypes.size()[1]
    sample_num = sample_embeddings.size()[0]  # n*k
    theta = torch.std(sample_embeddings)  # 获取方差
    sample_embdeddings = sample_embeddings.unsqueeze(0).repeat(150, 1, 1)  # [class_num, n*k, embed_size]
    prototypes = prototypes.unsqueeze(1).repeat(1, sample_num, 1)  # [class_num, n*k, embed_size]
    weight = torch.exp(
        -torch.pow(torch.norm(sample_embdeddings - prototypes, p=2, dim=2), exponent=2) / (2 * torch.pow(theta, 2)))
    return weight

# def drop_far_sample(prototypes, sample_embeddings):
#     theta = torch.std(sample_embeddings)
#     sample_num = sample_embeddings.size()[0]
#     newproto = prototypes.unsqueeze(0).repeat(sample_num, 1)
#     weight = torch.pow(torch.norm(sample_embeddings - newproto, p=2, dim=1), exponent=2)  # 获取了每个距离
#
#     return

def reweight_prototype(prototypes, sample_embeddings):
    # prototypes:[embed_size] 待重加权计算的原型
    # sample_embed:[sample_num, embed_size] 样本嵌入向量
    theta = torch.std(sample_embeddings)
    sample_num = sample_embeddings.size()[0]
    newproto = prototypes.unsqueeze(0).repeat(sample_num, 1)  # [sample_num, embed_size]
    # 两种方法 一种是指数加权平均  一种是使用 距离之和/距离 作为权重
    # 指数加权平均 weight:[sample_num]
    # weight = torch.exp(
    #     -torch.pow(torch.norm(sample_embeddings - newproto, p=2, dim=1), exponent=2) / (2 * torch.pow(theta, 2)))  # args.K_shot or 全部训练集
    # 换一种 使用 距离之和/距离 作为权重
    weight = torch.pow(torch.norm(sample_embeddings - newproto, p=2, dim=1), exponent=2) #获取了每个距离
    weight = weight.sum() / weight  # 距离之和/距离

    prototypes = (sample_embeddings * weight.reshape(sample_num, 1) / weight.sum()).sum(0) # [sample_num, embed_size] * [sample_num]
    return prototypes


def glove_cl_train(args):
    random.seed(0)
    with open(args.word2id_path) as f:
        word2id = json.load(f)
    embedding = Glove_Embedding(word2id, embedding_dim=args.embedding_dim, root=args.glove_path)
    encoder = CNN_Encoder(word_embedding=args.embedding_dim, hidden_size=args.encoder_dim, max_length=args.max_length)
    model = Embed_model(embedding, encoder)
    model = torch.load(args.model_save_path)
    fix_model = copy.deepcopy(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    model.to(device)
    tr_dataset = CLINC150_Glove(data_path=args.data_path, word2id=word2id, max_length=args.max_length, mode='cl_train')
    prototype_loader = DataLoader(tr_dataset, batch_size=100, shuffle=False)
    if args.losses == 'triplet':
        loss_fn = TripletLoss(args.margin)
    if args.losses == 'centertriplet':
        loss_fn = CenterTripletLoss()
    memory_size = args.class_num * args.K_shot
    memory = {'sentence': torch.zeros([memory_size, args.max_length]).long(),
              'label': torch.zeros(memory_size).long()}
    # 存储base类别样本
    for cls in range(0, 70):
        for idx, sample_idx in enumerate(random.sample(range(0, 100), args.K_shot)):
            memory['sentence'][cls * args.K_shot + idx], memory['label'][
                cls * args.K_shot + idx] = tr_dataset.__getitem__(cls * 100 + sample_idx)
    # 定义类原型
    prototypes = torch.Tensor(150, args.encoder_dim).float().to(device)
    # 计算原型
    model.eval()
    for sentence, label in prototype_loader:
        sentence = sentence.to(device)
        sample_embdeddings = model(sentence)
        prototypes[label[0].item()] = sample_embdeddings.mean(dim=0)
        # 循环重新加权计算原型
        for i in range(args.base_re_weight):
            prototypes[label[0].item()] = reweight_prototype(prototypes[label[0].item()], sample_embdeddings)  # 得到了每个类别原型及每个类别样本对应原型的权重。还需要限制样本。
    fix_prototypes = prototypes.clone()
    all_result = []
    for times in range(args.test_times):
        start_time = time.time()
        session_loss = [0.0] * 9
        session_acc = [0.0] * 9
        session_acc[0] = 0.9286
        session_data = Glove_session(args.data_path, word2id, args.max_length, 0, args.K_shot, 'cl_test')
        model = copy.deepcopy(fix_model)
        model = model.to(device)
        if args.cl_optim == 'sgd':
            cl_optimizer = torch.optim.SGD(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        elif args.cl_optim == 'adam':
            cl_optimizer = torch.optim.Adam(model.parameters(), lr=args.cl_lr, weight_decay=args.weight_decay)
        prototypes = fix_prototypes.clone()
        print('times:', times)
        for i in range(1, 9):
            print('#', i, end='')
            every_session_acc = [0.0] * 9
            # train
            session_data.set_session(i)
            session_dataloader = DataLoader(session_data, batch_size=args.batch_size, shuffle=False)
            if args.random_test:
                sentence, label = session_data.get_random_train_session()
            else:
                sentence, label = session_data.get_train_session()
            # 存储当前session数据
            memory_idx = (i - 1) * args.N_way * args.K_shot + 700
            memory['sentence'][memory_idx:args.N_way * args.K_shot + memory_idx], memory['label'][
                                                                                  memory_idx:args.N_way * args.K_shot + memory_idx] = sentence, label
            # to device
            sentence = sentence.to(device)
            label = label.to(device)
            if args.regularization == 'lwf':
                old_model = copy.deepcopy(model)
                old_model.eval()
                for paras in old_model.parameters():
                    paras.requires_grad = False
                old_out_put = old_model(sentence)
            if args.regularization == 'nl2':
                old_model = copy.deepcopy(model)
                old_model.eval()
                for paras in old_model.parameters():
                    paras.requires_grad = False
            if args.sdc != 'false':
                old_model = copy.deepcopy(model)
                old_model.eval()
            model.train()
            # 使用当前session数据微调
            for epoch in range(args.session_epoch):
                cl_optimizer.zero_grad()
                out_put = model(sentence)
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
            # 使用遇到所有类的数据重放（包含当前session数据）
            if args.replay:
                for j in range(args.replay_epoch):
                    old_sentence, old_label = torch.zeros([args.replay_batch, args.max_length]).long(), torch.zeros(
                        args.replay_batch).long()
                    # 抽取过去数据
                    for idx, sample_idx in enumerate(random.sample(range(0, memory_idx), args.replay_batch - 32)):
                        old_sentence[idx], old_label[idx] = memory['sentence'][sample_idx], memory['label'][sample_idx]
                    # 抽取当前session数据
                    for idx, sample_idx in enumerate(
                            random.sample(range(memory_idx, memory_idx + args.K_shot * args.N_way), 32)):
                        old_sentence[idx + args.replay_batch - 32], old_label[idx + args.replay_batch - 32] = \
                            memory['sentence'][sample_idx], memory['label'][sample_idx]
                    old_sentence, old_label = old_sentence.to(device), old_label.to(device)
                    cl_optimizer.zero_grad()
                    logits = model(old_sentence)
                    loss = loss_fn(logits, old_label)
                    loss.backward()
                    cl_optimizer.step()
            # session_loss[i] = round(session_loss[i] / (args.session_epoch + 0.0001), 4)
            # 在所有遇见过的类上测试
            correct = 0
            with torch.no_grad():
                # 得到新的类的原型
                model.eval()
                out_put = model(sentence)
                # 语义漂移补偿
                if args.sdc == 'all':
                    old_memory_output = old_model(memory['sentence'][:args.N_way * args.K_shot + memory_idx].to(device))
                    new_memory_output = model(memory['sentence'][:args.N_way * args.K_shot + memory_idx].to(device))
                    weight = compute_sample_weight(prototypes, old_memory_output)
                    for cls in range(0, 70 + (i - 1) * args.N_way):
                        compensation = ((new_memory_output - old_memory_output) * weight[cls].reshape(
                            args.N_way * args.K_shot + memory_idx, 1) / weight[cls].sum()).sum(dim=0)
                        prototypes[cls] = prototypes[cls] + compensation
                elif args.sdc == 'base':
                    old_memory_output = old_model(memory['sentence'][:args.N_way * args.K_shot + memory_idx].to(device))
                    new_memory_output = model(memory['sentence'][:args.N_way * args.K_shot + memory_idx].to(device))
                    weight = compute_sample_weight(prototypes, old_memory_output)
                    for cls in range(0, 70):
                        compensation = ((new_memory_output - old_memory_output) * weight[cls].reshape(
                            args.N_way * args.K_shot + memory_idx, 1) / weight[cls].sum()).sum(dim=0)
                        prototypes[cls] = prototypes[cls] + compensation
                # 重新计算类原型
                if args.recompute_proto == 'all':
                    for cls in range(0, 70 + (i - 1) * args.N_way):
                        old_sentence = memory['sentence'][cls * args.K_shot:(cls + 1) * args.K_shot].to(device)
                        sample_embdeddings = model(old_sentence)
                        prototypes[cls] = sample_embdeddings.mean(dim=0)
                        for t in range(args.incremental_re_weight):
                            prototypes[cls] = reweight_prototype(prototypes[cls], sample_embdeddings)
                elif args.recompute_proto == 'new':
                    for cls in range(70, 70 + (i - 1) * args.N_way):
                        old_sentence = memory['sentence'][cls * args.K_shot:(cls + 1) * args.K_shot].to(device)
                        prototypes[cls] = model(old_sentence).mean(dim=0)
                        for t in range(args.incremental_re_weight):
                            prototypes[cls] = reweight_prototype(prototypes[cls], sample_embdeddings)
                out_put = out_put.view(10, args.K_shot, args.encoder_dim)
                new_class_proto = out_put.mean(dim=1)
                for cls in range(args.N_way):
                    for t in range(args.incremental_re_weight):
                        new_class_proto[cls] = reweight_prototype(new_class_proto[cls], out_put[cls])
                prototypes[i * 10 + 60:i * 10 + 70] = new_class_proto
                # 在所有遇见过的类上测试
                # print(len(session_data), ' ', len(session_dataloader))
                for batch_idx, (sentence, label) in enumerate(session_dataloader):
                    sentence = sentence.to(device)
                    label = label.to(device)
                    dist = euclidean_dist(model(sentence), prototypes[:i * 10 + 70])
                    _, pred = torch.max(-dist, dim=-1)
                    correct += (pred == label).sum().float().item()
                    correct_num = (pred == label).sum().float().item()
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
        # print('用时:', (time.time() - start_time) / 60)
        # print('loss:', session_loss)
        all_result.append(session_acc)
    return all_result


def base_train(args, ):
    return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', help='数据路径', type=str,
                           default='D:/code/dataset/CLINC150/data_full.json')
    argparser.add_argument('--glove_path', type=str, default='D:/code/pretrained_model/glove.6B')
    argparser.add_argument('--batch_size', help='批次大小', type=int,
                           default=300)
    argparser.add_argument('--replay_batch', help='批次大小', type=int,
                           default=256)
    argparser.add_argument('--word2id_path', type=str,
                           default='word2id.json')
    argparser.add_argument('--weight_decay', help='', type=float, default=0)
    argparser.add_argument('--class_num', help='', type=int, default=150)
    argparser.add_argument('--embedding_dim', type=int, default=100)
    argparser.add_argument('--encoder_dim', type=int, default=512)
    argparser.add_argument('--max_length', type=int, default=28)
    argparser.add_argument('--margin', help='triplet loss margin', type=float, default=0.5)
    argparser.add_argument('--K_shot', help='每类样本数量', type=int, default=10)
    argparser.add_argument('--N_way', help='每个session类别数量', type=int, default=10)
    argparser.add_argument('--reuse_base', help='是否使用预训练好的度量网络', type=bool, default=True)
    argparser.add_argument('--result_path', help='训练结果存储路径', type=str,
                           default='clinc_result/clinc_base_glove_ftall_triplet_m0.5_decay_adam1e-3.txt')
    argparser.add_argument('--model_save_path', help='model存储路径', type=str,
                           default='clinc_result/clinc_base_glove_ftall_triplet_m0.5_decay_adam1e-3.pth')
    argparser.add_argument('--regularization', help='使用什么正则化 比如lwf', type=str, default='nolwf')
    argparser.add_argument('--r_xishu', help='正则化系数', type=float, default=0.005)
    argparser.add_argument('--cl_lr', help='增量学习学习率', type=float,
                           default=5e-5)
    argparser.add_argument('--cl_optim', help='优化器', type=str, default='adam')
    # 可选参数为all、new、false
    argparser.add_argument('--recompute_proto', help='是否重计算原型', type=str, default='new')
    # 可选参数为all、base、false
    argparser.add_argument('--sdc', help='是否补偿语义漂移', type=str, default='base')
    argparser.add_argument('--replay', help='是否重放', type=bool, default=True)
    argparser.add_argument('--session_epoch', help='增量任务轮次', type=int, default=0)
    argparser.add_argument('--replay_epoch', help='重放任务轮次', type=int, default=10)
    argparser.add_argument('--optimizer', help='使用哪种优化器', type=str, default='adam')
    argparser.add_argument('--losses', help='损失函数', type=str, default='triplet')
    argparser.add_argument('--test_times', help='测试次数', type=int, default=3)
    argparser.add_argument('--random_test', help='是否随机抽取session数据', type=bool, default=True)
    argparser.add_argument('--incremental_re_weight', help='重加权计算原型次数_针对增量session', type=int, default=1)
    argparser.add_argument('--base_re_weight', help='重加权计算原型次数_针对base_session', type=int, default=0)
    args = argparser.parse_args()
    all_result = glove_cl_train(args=args)
    if args.test_times > 1:
        print('avg result:')
        print(torch.Tensor(all_result).mean(dim=0))
    # print('重放次数：', args.replay_epoch, '微调次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：',
    #       args.r_xishu, ' 学习率：', args.cl_lr,
    #       ' 优化器：', args.cl_optim, ' sdc:',args.sdc,' recompute:',args.recompute_proto)
    print(args)
    print('*******************************************************************************')
    # args.recompute_proto = 'all'
    # args.sdc = 'false'
    # all_result = glove_cl_train(args=args)
    # if args.test_times > 1:
    #     print('avg result:')
    #     print(torch.Tensor(all_result).mean(dim=0))
    # print('重放次数：', args.replay_epoch, '微调次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：',
    #       args.r_xishu, ' 学习率：', args.cl_lr,
    #       ' 优化器：', args.cl_optim, ' sdc:',args.sdc,' recompute:',args.recompute_proto)
    # print('*******************************************************************************')
    # args.recompute_proto = 'false'
    # args.sdc = 'all'
    # all_result = glove_cl_train(args=args)
    # if args.test_times > 1:
    #     print('avg result:')
    #     print(torch.Tensor(all_result).mean(dim=0))
    # print('重放次数：', args.replay_epoch, '微调次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：',
    #       args.r_xishu, ' 学习率：', args.cl_lr,
    #       ' 优化器：', args.cl_optim, ' sdc:', args.sdc, ' recompute:', args.recompute_proto)
    # print('*******************************************************************************')
    # args.recompute_proto = 'new'
    # args.sdc = 'base'
    # args.session_epoch = 0
    # args.replay_epoch = 50
    # all_result = glove_cl_train(args=args)
    # if args.test_times > 1:
    #     print('avg result:')
    #     print(torch.Tensor(all_result).mean(dim=0))
    # print('重放次数：', args.replay_epoch, '微调次数：', args.session_epoch, ' 正则化：', args.regularization,
    #       '  正则化系数：',
    #       args.r_xishu, ' 学习率：', args.cl_lr,
    #       ' 优化器：', args.cl_optim, ' sdc:', args.sdc, ' recompute:', args.recompute_proto)
    # print('*******************************************************************************')
    # args.recompute_proto = 'all'
    # args.sdc = 'false'
    # args.session_epoch = 10
    # args.replay_epoch = 40
    # all_result = glove_cl_train(args=args)
    # if args.test_times > 1:
    #     print('avg result:')
    #     print(torch.Tensor(all_result).mean(dim=0))
    # print('重放次数：', args.replay_epoch, '微调次数：', args.session_epoch, ' 正则化：', args.regularization, '  正则化系数：',
    #       args.r_xishu, ' 学习率：', args.cl_lr,
    #       ' 优化器：', args.cl_optim, ' sdc:', args.sdc, ' recompute:', args.recompute_proto)
    # print('*******************************************************************************')
