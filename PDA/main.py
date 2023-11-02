from __future__ import print_function
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import torch.utils.data as Data
from metrics import quadratic_weighted_kappa
from options import args
from reader import *
import resnet as models
device = torch.device("cuda:0")
momentum = 0.9
l2_decay = 5e-4
def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    source1_id = args.source1_id
    source2_id = args.source2_id
    source3_id = args.source3_id
    source4_id = args.source4_id
    source5_id = args.source5_id
    source6_id = args.source6_id
    source7_id = args.source7_id
    devprompt_id = args.devprompt_id
    embedding_path = args.embedding_dict
    oov = args.oov
    embedding = args.embedding
    embedd_dim = args.embedding_dim

    count = []
    qwkfilev=open("images/modelnew/tqwk6.txt", "w")
    for epoch in range(1):
        source1_path = os.path.join('newdata', f'{source1_id}.task.train')
        source2_path = os.path.join('newdata', f'{source2_id}.task.train')
        source3_path = os.path.join('newdata', f'{source3_id}.task.train')
        source4_path = os.path.join('newdata', f'{source4_id}.task.train')
        source5_path = os.path.join('newdata', f'{source5_id}.task.train')
        source6_path = os.path.join('newdata', f'{source6_id}.task.train')
        source7_path = os.path.join('newdata', f'{source7_id}.task.train')
        dev_path = os.path.join('newdata', f'{devprompt_id}.task.dev')
        test_path = os.path.join('newdata', f'{devprompt_id}.task.test')
        datapaths = [source1_path, source2_path, source3_path, source4_path, source5_path, source6_path, source7_path,
                     dev_path, test_path]
        print(datapaths[:7])
        vocab = create_vocab(file_path=datapaths[:7], source1_id=source1_id, source2_id=source2_id,
                             source3_id=source3_id, source4_id=source4_id, source5_id=source5_id, source6_id=source6_id,
                             source7_id=source7_id, vocab_size=0, tokenize_text=True, to_lower=True)
        (X_source1, Y_source1, mask_source1, source1_pmt,essay1_ids), (X_source2, Y_source2, mask_source2, source2_pmt,essay2_ids), \
        (X_source3, Y_source3, mask_source3, source3_pmt,essay3_ids), (X_source4, Y_source4, mask_source4, source4_pmt,essay4_ids), \
        (X_source5, Y_source5, mask_source5, source5_pmt,essay5_ids), (X_source6, Y_source6, mask_source6, source6_pmt,essay6_ids), \
        (X_source7, Y_source7, mask_source7, source7_pmt,essay7_ids), (X_dev, Y_dev, mask_dev, dev_pmt,essaydev_ids), (
        X_test, Y_test, mask_test, test_pmt,essaytest_ids), \
        embed_table, overal_maxlen, overal_maxnum = prepare_sentence_data(embedd_dim, source1_id, source2_id,
                                                                          source3_id, source4_id, source5_id,
                                                                          source6_id, source7_id, devprompt_id,
                                                                          datapaths, vocab, embedding_path, embedding)
        # print(embed_table.shape)
        max_sentnum = overal_maxnum
        max_sentlen = overal_maxlen
        Y_source1 = torch.tensor(Y_source1)
        Y_source2 = torch.tensor(Y_source2)
        Y_source3 = torch.tensor(Y_source3)
        Y_source4 = torch.tensor(Y_source4)
        Y_source5 = torch.tensor(Y_source5)
        Y_source6 = torch.tensor(Y_source6)
        Y_source7 = torch.tensor(Y_source7)
        Y_dev = torch.tensor(Y_dev)
        Y_test = torch.tensor(Y_test)
        X_source1 = torch.LongTensor(X_source1)
        X_source2 = torch.LongTensor(X_source2)
        X_source3 = torch.LongTensor(X_source3)
        X_source4 = torch.LongTensor(X_source4)
        X_source5 = torch.LongTensor(X_source5)
        X_source6 = torch.LongTensor(X_source6)
        X_source7 = torch.LongTensor(X_source7)
        X_dev = torch.LongTensor(X_dev)
        X_test = torch.LongTensor(X_test)
        essay1_ids = torch.LongTensor(essay1_ids)
        essay2_ids = torch.LongTensor(essay2_ids)
        essay3_ids = torch.LongTensor(essay3_ids)
        essay4_ids = torch.LongTensor(essay4_ids)
        essay5_ids = torch.LongTensor(essay5_ids)
        essay6_ids = torch.LongTensor(essay6_ids)
        essay7_ids = torch.LongTensor(essay7_ids)
        essaydev_ids = torch.LongTensor(essaydev_ids)
        essaytest_ids = torch.LongTensor(essaytest_ids)
        source1_pmt = torch.LongTensor(source1_pmt)
        source2_pmt = torch.LongTensor(source2_pmt)
        source3_pmt = torch.LongTensor(source3_pmt)
        source4_pmt = torch.LongTensor(source4_pmt)
        source5_pmt = torch.LongTensor(source5_pmt)
        source6_pmt = torch.LongTensor(source6_pmt)
        source7_pmt = torch.LongTensor(source7_pmt)
        dev_pmt = torch.LongTensor(dev_pmt)
        test_pmt = torch.LongTensor(test_pmt)
        source1_data = Data.TensorDataset(X_source1, Y_source1, source1_pmt,essay1_ids)
        source2_data = Data.TensorDataset(X_source2, Y_source2, source2_pmt,essay2_ids)
        source3_data = Data.TensorDataset(X_source3, Y_source3, source3_pmt,essay3_ids)
        source4_data = Data.TensorDataset(X_source4, Y_source4, source4_pmt,essay4_ids)
        source5_data = Data.TensorDataset(X_source5, Y_source5, source5_pmt,essay5_ids)
        source6_data = Data.TensorDataset(X_source6, Y_source6, source6_pmt,essay6_ids)
        source7_data = Data.TensorDataset(X_source7, Y_source7, source7_pmt,essay7_ids)
        dev_data = Data.TensorDataset(X_dev, Y_dev, dev_pmt,essaydev_ids)
        test_data = Data.TensorDataset(X_test, Y_test, test_pmt,essaytest_ids)
        source1_loader = Data.DataLoader(dataset=source1_data, batch_size=batch_size, shuffle=True)
        source2_loader = Data.DataLoader(dataset=source2_data, batch_size=batch_size, shuffle=True)
        source3_loader = Data.DataLoader(dataset=source3_data, batch_size=batch_size, shuffle=True)
        source4_loader = Data.DataLoader(dataset=source4_data, batch_size=batch_size, shuffle=True)
        source5_loader = Data.DataLoader(dataset=source5_data, batch_size=batch_size, shuffle=True)
        source6_loader = Data.DataLoader(dataset=source6_data, batch_size=batch_size, shuffle=True)
        source7_loader = Data.DataLoader(dataset=source7_data, batch_size=batch_size, shuffle=True)
        dev_loader = Data.DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
        # target_data = Data.ConcatDataset([dev_data, test_data])
        target_train_loader = Data.DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
        model = models.MFSAN(batch_size, embed_table, max_sentnum, max_sentlen)
        print(model)
        if torch.cuda.is_available():
            model.to(device)
        best_qwk_t = 0.0
        best_loss_t = 0.0
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # …………………………………………………………………………………atdoc_na…………………………………………………………………………
        mem_fea = torch.rand(len(target_train_loader.dataset), 100).to(device)
        mem_fea1 = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        mem_cls1 = torch.ones(len(target_train_loader.dataset), 4).to(device) / 4
        mem_fea2,mem_fea3,mem_fea4,mem_fea5,mem_fea6,mem_fea7=mem_fea1,mem_fea1,mem_fea1,mem_fea1,mem_fea1,mem_fea1
        mem_cls2,mem_cls3,mem_cls4,mem_cls5,mem_cls6,mem_cls7=mem_cls1,mem_cls1,mem_cls1,mem_cls1,mem_cls1,mem_cls1

        len_dataloader = max(len(source1_loader), len(source2_loader), len(source3_loader), len(source4_loader),
                             len(source5_loader), len(source6_loader), len(source7_loader), len(target_train_loader))
        num_iter = len_dataloader * num_epochs

        tab = 0
        for i in range(1, num_iter + 1):
            model.train()
            decay = math.pow((1 + 10 * (i - 1) / (num_iter)), 0.75)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate / decay
                param_group['weight_decay'] = 5e-4
                param_group['momentum'] = 0.9
            source1_iter = iter(source1_loader)
            source2_iter = iter(source2_loader)
            source3_iter = iter(source3_loader)
            source4_iter = iter(source4_loader)
            source5_iter = iter(source5_loader)
            source6_iter = iter(source6_loader)
            source7_iter = iter(source7_loader)
            target_iter = iter(target_train_loader)
            try:
                data_target = target_iter.next()
                target_data, _,_,idx= data_target
                t_domain_label = torch.ones(batch_size, 1)
                if len(target_data) < batch_size:
                    data_target = target_iter.next()
                    target_data, _,_,idx = data_target
                    t_domain_label = torch.ones(batch_size, 1)
            except Exception as err:
                target_iter = iter(target_train_loader)
                data_target = target_iter.next()
                target_data, _,_,idx= data_target
                t_domain_label = torch.ones(batch_size, 1)
                if len(target_data) < batch_size:
                    data_target = target_iter.next()
                    target_data, _,_,idx= data_target
                    t_domain_label = torch.ones(batch_size, 1)
            try:
                data_source = source1_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source1_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            except Exception as err:
                source1_iter = iter(source1_loader)
                data_source = source1_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source1_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            if torch.cuda.is_available():
                source_data = source_data.to(device)
                target_data = target_data.to(device)
                idx=idx.to(device)
                source_label = torch.flatten(source_label)
                source_label = source_label.to(device)
                s_domain_label = s_domain_label.to(device)
                t_domain_label = t_domain_label.to(device)
            # print(source_label)----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # Tensor是Pytorch的一个完美组件(可以生成高维数组)，但是要构建神经网络还是远远不够的，我们需要能够计算图的Tensor，那就是Variable。
            # Variable是对Tensor的一个封装，操作和Tensor是一样的，但是每个Variable都有三个属性，
            # .data得到该对象的Tensor数值，通过.grad_fn知道其是通过什么方式如SumBackward方式等得到的，通过.grad得到了x和y的梯度
            source_data, source_label = Variable(source_data), Variable(source_label)
            target_data = Variable(target_data)
            s_domain_label = Variable(s_domain_label)
            t_domain_label = Variable(t_domain_label)
            # 每一次先将梯度清0
            optimizer.zero_grad()
            cls_loss, domain_loss, l1_loss,pl1_loss, cls_loss_total, features_source1, pred1, we1,outputs_target1, data_src1 = model(
                source_data,
                target_data,
                source_label,
                s_domain_label,
                t_domain_label, mem_fea1,mem_fea2,mem_fea3,mem_fea4,mem_fea5,mem_fea6,mem_fea7,
                mem_cls1,mem_cls2,mem_cls3,mem_cls4,mem_cls5,mem_cls6,mem_cls7,idx,
                max_iter=num_iter,
                mark=1)
            # 没有实例权重用下面两句
            # outputs_target1 = F.log_softmax(outputs_target1, dim=1)
            # classifier_loss1 = F.nll_loss(outputs_target1, pred1)
            # 用到实例的权重
            outputs_target1 = F.log_softmax(outputs_target1, dim=1)
            classifier_loss1 = F.nll_loss(outputs_target1, pred1)
            classifier_loss1 = torch.sum(we1 * classifier_loss1) / (torch.sum(we1).item())
            gamma = 2 / (1 + math.exp(-10 * (i) / num_iter)) - 1
            ga=(i/num_iter)*10
            #所有损失都在
            # loss = cls_loss + gamma * (domain_loss + l1_loss + 1 * (pl1_loss+classifier_loss1 + cls_loss_total))
            # 参数敏感性  2:x时
            loss = (cls_loss +gamma*(
                        1 * (domain_loss ) + 0.5* (classifier_loss1 + cls_loss_total)))+ l1_loss


            loss.backward()
            optimizer.step()
            print('Train source1 iter: {}\tLoss: {:.6f}'.format(i, loss.item()))
            try:
                # source_data, source_label = source2_iter.next()
                data_source = source2_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source2_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            except Exception as err:
                source2_iter = iter(source2_loader)
                data_source = source2_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source2_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            if torch.cuda.is_available():
                source_data = source_data.to(device)
                # target_data2 = target_data2.to(device)
                source_label = torch.flatten(source_label)
                source_label = source_label.to(device)
                s_domain_label = s_domain_label.to(device)
                # t_domain_label = t_domain_label.to(device)
            source_data, source_label = Variable(source_data), Variable(source_label)
            # target_data2 = Variable(target_data2)
            s_domain_label = Variable(s_domain_label)
            # t_domain_label = Variable(t_domain_label)
            optimizer.zero_grad()
            cls_loss, domain_loss, l1_loss,pl1_loss, cls_loss_total, features_source2, pred2,we2, outputs_target2, data_src2 = model(
                source_data,
                target_data,
                source_label,
                s_domain_label,
                t_domain_label, mem_fea1,mem_fea2,mem_fea3,mem_fea4,mem_fea5,mem_fea6,mem_fea7,
                mem_cls1,mem_cls2,mem_cls3,mem_cls4,mem_cls5,mem_cls6,mem_cls7,idx,max_iter=num_iter,
                mark=2)
            # 没有实例权重用下面两句
            # outputs_target2 = F.log_softmax(outputs_target2, dim=1)
            # classifier_loss2 = F.nll_loss(outputs_target2, pred2)
            # 用到实例的权重
            outputs_target2 = F.log_softmax(outputs_target2, dim=1)
            classifier_loss2 = F.nll_loss(outputs_target2, pred2)
            classifier_loss2 = torch.sum(we2 * classifier_loss2) / (torch.sum(we2).item())
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            gamma = 2 / (1 + math.exp(-10 * (i) / num_iter)) - 1
            ga = (i / num_iter) * 10
            #所有损失都在   整体：对齐=2:1
            # loss = cls_loss + gamma * (domain_loss + l1_loss + 1 * (pl1_loss+classifier_loss2 + cls_loss_total))

            # 参数敏感性6  2:x时
            loss = (cls_loss + gamma*(1*(domain_loss )+ 0.5* (classifier_loss2 + cls_loss_total)))+ l1_loss

            loss.backward()
            optimizer.step()
            print('Train source2 iter: {}\tLoss: {:.6f}'.format(i, loss.item()))

            try:
                # 得到源域3的数据和标签以及目标域的数据
                # source_data, source_label = source1_iter.next()
                data_source = source3_iter.next()
                source_data, source_label, prompt_label,_= data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source3_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            except Exception as err:
                source3_iter = iter(source3_loader)
                data_source = source3_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source3_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            if torch.cuda.is_available():
                source_data = source_data.to(device)
                # target_data3 = target_data3.to(device)
                source_label = torch.flatten(source_label)
                source_label = source_label.to(device)
                s_domain_label = s_domain_label.to(device)
                # t_domain_label = t_domain_label.to(device)
            # print(source_label)----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # Tensor是Pytorch的一个完美组件(可以生成高维数组)，但是要构建神经网络还是远远不够的，我们需要能够计算图的Tensor，那就是Variable。
            # Variable是对Tensor的一个封装，操作和Tensor是一样的，但是每个Variable都有三个属性，
            # .data得到该对象的Tensor数值，通过.grad_fn知道其是通过什么方式如SumBackward方式等得到的，通过.grad得到了x和y的梯度
            source_data, source_label = Variable(source_data), Variable(source_label)
            # print(source_label)-----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # target_data3 = Variable(target_data3)
            s_domain_label = Variable(s_domain_label)
            # t_domain_label = Variable(t_domain_label)
            # 每一次先将梯度清0
            optimizer.zero_grad()
            # 计算分类损失，mmd损失和差异损失
            cls_loss, domain_loss, l1_loss,pl1_loss, cls_loss_total, features_source3, pred3,we3, outputs_target3, data_src3 = model(
                source_data,
                target_data,
                source_label,
                s_domain_label,
                t_domain_label, mem_fea1,mem_fea2,mem_fea3,mem_fea4,mem_fea5,mem_fea6,mem_fea7,
                mem_cls1,mem_cls2,mem_cls3,mem_cls4,mem_cls5,mem_cls6,mem_cls7,idx,max_iter=num_iter,
                mark=3)
            # 没有实例权重用下面两句
            # outputs_target3 = F.log_softmax(outputs_target3, dim=1)
            # classifier_loss3 = F.nll_loss(outputs_target3, pred3)
            # 用到实例的权重
            outputs_target3 = F.log_softmax(outputs_target3, dim=1)
            classifier_loss3 = F.nll_loss(outputs_target3, pred3)
            classifier_loss3 = torch.sum(we3 * classifier_loss3) / (torch.sum(we3).item())
            gamma = 2 / (1 + math.exp(-10 * (i) / num_iter)) - 1
            ga = (i / num_iter) * 10
            #所有损失都在
            # loss = cls_loss + gamma * (domain_loss + l1_loss + 1 * (pl1_loss+classifier_loss3 + cls_loss_total))

            # 参数敏感性6  2:x时
            loss = (cls_loss +gamma*(1*(domain_loss )+ 0.5* (classifier_loss3 + cls_loss_total)))+ l1_loss

            loss.backward()
            optimizer.step()
            print('Train source3 iter: {}\tLoss: {:.6f}'.format(i, loss.item()))
            try:
                # 得到源域1的数据和标签以及目标域的数据
                # source_data, source_label = source1_iter.next()
                data_source = source4_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source4_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            except Exception as err:
                source4_iter = iter(source4_loader)
                data_source = source4_iter.next()
                source_data, source_label, prompt_label,_= data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source4_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            if torch.cuda.is_available():
                source_data = source_data.to(device)
                # target_data4 = target_data4.to(device)
                source_label = torch.flatten(source_label)
                source_label = source_label.to(device)
                s_domain_label = s_domain_label.to(device)
                # t_domain_label = t_domain_label.to(device)
            # print(source_label)----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # Tensor是Pytorch的一个完美组件(可以生成高维数组)，但是要构建神经网络还是远远不够的，我们需要能够计算图的Tensor，那就是Variable。
            # Variable是对Tensor的一个封装，操作和Tensor是一样的，但是每个Variable都有三个属性，
            # .data得到该对象的Tensor数值，通过.grad_fn知道其是通过什么方式如SumBackward方式等得到的，通过.grad得到了x和y的梯度
            source_data, source_label = Variable(source_data), Variable(source_label)
            # print(source_label)-----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # target_data4 = Variable(target_data4)
            s_domain_label = Variable(s_domain_label)
            # t_domain_label = Variable(t_domain_label)
            # 每一次先将梯度清0
            optimizer.zero_grad()
            # 计算分类损失，mmd损失和差异损失
            cls_loss, domain_loss, l1_loss, pl1_loss,cls_loss_total, features_source4, pred4,we4, outputs_target4, data_src4 = model(
                source_data,
                target_data,
                source_label,
                s_domain_label,
                t_domain_label, mem_fea1,mem_fea2,mem_fea3,mem_fea4,mem_fea5,mem_fea6,mem_fea7,
                mem_cls1,mem_cls2,mem_cls3,mem_cls4,mem_cls5,mem_cls6,mem_cls7,idx,max_iter=num_iter,
                mark=4)
            # 没有实例权重用下面两句
            # outputs_target4 = F.log_softmax(outputs_target4, dim=1)
            # classifier_loss4 = F.nll_loss(outputs_target4, pred4)
            # 用到实例的权重
            outputs_target4 = F.log_softmax(outputs_target4, dim=1)
            classifier_loss4 = F.nll_loss(outputs_target4, pred4)
            classifier_loss4 = torch.sum(we4 * classifier_loss4) / (torch.sum(we4).item())
            gamma = 2 / (1 + math.exp(-10 * (i) / num_iter)) - 1
            ga = (i / num_iter) * 10


            # 参数敏感性6  2:x时
            loss = (cls_loss + gamma*(1*(domain_loss )+ 0.5* (classifier_loss4 + cls_loss_total)))+ l1_loss

            loss.backward()
            optimizer.step()
            print('Train source4 iter: {}\tLoss: {:.6f}'.format(i, loss.item()))

            try:
                # 得到源域1的数据和标签以及目标域的数据
                # source_data, source_label = source1_iter.next()
                data_source = source5_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source5_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            except Exception as err:
                source5_iter = iter(source5_loader)
                data_source = source5_iter.next()
                source_data, source_label, prompt_label,_= data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source5_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            if torch.cuda.is_available():
                source_data = source_data.to(device)
                # target_data5 = target_data5.to(device)
                source_label = torch.flatten(source_label)
                source_label = source_label.to(device)
                s_domain_label = s_domain_label.to(device)
                # t_domain_label = t_domain_label.to(device)
            # print(source_label)----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # Tensor是Pytorch的一个完美组件(可以生成高维数组)，但是要构建神经网络还是远远不够的，我们需要能够计算图的Tensor，那就是Variable。
            # Variable是对Tensor的一个封装，操作和Tensor是一样的，但是每个Variable都有三个属性，
            # .data得到该对象的Tensor数值，通过.grad_fn知道其是通过什么方式如SumBackward方式等得到的，通过.grad得到了x和y的梯度
            source_data, source_label = Variable(source_data), Variable(source_label)
            # print(source_label)-----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # target_data5 = Variable(target_data5)
            s_domain_label = Variable(s_domain_label)
            # t_domain_label = Variable(t_domain_label)
            # 每一次先将梯度清0
            optimizer.zero_grad()
            # 计算分类损失，mmd损失和差异损失
            cls_loss, domain_loss, l1_loss, pl1_loss,cls_loss_total, features_source5, pred5,we5, outputs_target5, data_src5 = model(
                source_data,
                target_data,
                source_label,
                s_domain_label,
                t_domain_label, mem_fea1,mem_fea2,mem_fea3,mem_fea4,mem_fea5,mem_fea6,mem_fea7,
                mem_cls1,mem_cls2,mem_cls3,mem_cls4,mem_cls5,mem_cls6,mem_cls7,idx,max_iter=num_iter,
                mark=5)
            # 没有实例权重用下面两句
            # outputs_target5 = F.log_softmax(outputs_target5, dim=1)
            # classifier_loss5 = F.nll_loss(outputs_target5, pred5)
            # 用到实例的权重
            outputs_target5 = F.log_softmax(outputs_target5, dim=1)
            classifier_loss5 = F.nll_loss(outputs_target5, pred5)
            classifier_loss5 = torch.sum(we5 * classifier_loss5) / (torch.sum(we5).item())
            gamma = 2 / (1 + math.exp(-10 * (i) / num_iter)) - 1
            #所有损失都在
            # loss = cls_loss + gamma * (domain_loss + l1_loss + 1 * (pl1_loss+classifier_loss5 + cls_loss_total))
            ga = (i / num_iter) * 10

            # 参数敏感性6  2:x时
            loss = (cls_loss +gamma*(1*(domain_loss )+ 0.5* (classifier_loss5 + cls_loss_total)))+ l1_loss

            loss.backward()
            optimizer.step()
            print('Train source5 iter: {}\tLoss: {:.6f}'.format(i, loss.item()))
            try:
                # 得到源域1的数据和标签以及目标域的数据
                # source_data, source_label = source1_iter.next()
                data_source = source6_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source6_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            except Exception as err:
                source6_iter = iter(source6_loader)
                data_source = source6_iter.next()
                source_data, source_label, prompt_label,_ = data_source
                s_domain_label = torch.zeros(batch_size, 1)
                if len(source_label) < batch_size:
                    data_source = source6_iter.next()
                    source_data, source_label, prompt_label,_ = data_source
                    s_domain_label = torch.zeros(batch_size, 1)
            if torch.cuda.is_available():
                source_data = source_data.to(device)
                # target_data6 = target_data6.to(device)
                source_label = torch.flatten(source_label)
                source_label = source_label.to(device)
                s_domain_label = s_domain_label.to(device)
                # t_domain_label = t_domain_label.to(device)
            # print(source_label)----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # Tensor是Pytorch的一个完美组件(可以生成高维数组)，但是要构建神经网络还是远远不够的，我们需要能够计算图的Tensor，那就是Variable。
            # Variable是对Tensor的一个封装，操作和Tensor是一样的，但是每个Variable都有三个属性，
            # .data得到该对象的Tensor数值，通过.grad_fn知道其是通过什么方式如SumBackward方式等得到的，通过.grad得到了x和y的梯度
            source_data, source_label = Variable(source_data), Variable(source_label)
            # print(source_label)-----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # target_data6 = Variable(target_data6)
            s_domain_label = Variable(s_domain_label)
            # t_domain_label = Variable(t_domain_label)
            # 每一次先将梯度清0
            optimizer.zero_grad()
            # 计算分类损失，mmd损失和差异损失
            cls_loss, domain_loss, l1_loss,pl1_loss, cls_loss_total, features_source6, pred6,we6, outputs_target6, data_src6 = model(
                source_data,
                target_data,
                source_label,
                s_domain_label,
                t_domain_label, mem_fea1,mem_fea2,mem_fea3,mem_fea4,mem_fea5,mem_fea6,mem_fea7,
                mem_cls1,mem_cls2,mem_cls3,mem_cls4,mem_cls5,mem_cls6,mem_cls7,idx,max_iter=num_iter,
                mark=6)
            # 没有实例权重用下面两句
            # outputs_target7 = F.log_softmax(outputs_target7, dim=1)
            # classifier_loss7 = F.nll_loss(outputs_target7, pred7)
            # 用到实例的权重
            outputs_target6= F.log_softmax(outputs_target6, dim=1)
            classifier_loss6 = F.nll_loss(outputs_target6, pred6)
            classifier_loss6 = torch.sum(we6 * classifier_loss6) / (torch.sum(we6).item())
            gamma = 2 / (1 + math.exp(-10 * (i) / num_iter)) - 1
            ga = (i / num_iter) * 10

            # 参数敏感性6  2:x时
            loss = (cls_loss +  gamma*(1 * (domain_loss ) + 0.5* (classifier_loss6 + cls_loss_total)))+ l1_loss

            loss.backward()
            optimizer.step()
            print('Train source6 iter: {}\tLoss: {:.6f}'.format(i, loss.item()))
            try:
                # 得到源域1的数据和标签以及目标域的数据
                # source_data, source_label = source1_iter.next()
                data_source7 = source7_iter.next()
                source_data7, source_label7, prompt_label7,_ = data_source7
                s_domain_label7 = torch.zeros(batch_size, 1)
                if len(source_label7) < batch_size:
                    data_source7 = source7_iter.next()
                    source_data7, source_label7, prompt_label7,_ = data_source7
                    s_domain_label7 = torch.zeros(batch_size, 1)
            except Exception as err:
                source7_iter = iter(source7_loader)
                data_source7 = source7_iter.next()
                source_data7, source_label7, prompt_label7,_ = data_source7
                s_domain_label7 = torch.zeros(batch_size, 1)
                if len(source_label7) < batch_size:
                    data_source7 = source7_iter.next()
                    source_data7, source_label7, prompt_label7,_ = data_source7
                    s_domain_label7 = torch.zeros(batch_size, 1)
            if torch.cuda.is_available():
                source_data7 = source_data7.to(device)
                # target_data7 = target_data7.to(device)
                source_label7 = torch.flatten(source_label7)
                source_label7 = source_label7.to(device)
                s_domain_label7 = s_domain_label7.to(device)
                # t_domain_label= t_domain_label.to(device)
            # print(source_label)----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # Tensor是Pytorch的一个完美组件(可以生成高维数组)，但是要构建神经网络还是远远不够的，我们需要能够计算图的Tensor，那就是Variable。
            # Variable是对Tensor的一个封装，操作和Tensor是一样的，但是每个Variable都有三个属性，
            # .data得到该对象的Tensor数值，通过.grad_fn知道其是通过什么方式如SumBackward方式等得到的，通过.grad得到了x和y的梯度
            source_data7, source_label7 = Variable(source_data7), Variable(source_label7)
            # print(source_label)-----tensor([2, 3, 2, 2, 2, 2, 3, 3], device='cuda:0')
            # target_data7 = Variable(target_data7)
            s_domain_label7 = Variable(s_domain_label7)
            # t_domain_label = Variable(t_domain_label)
            # 每一次先将梯度清0
            optimizer.zero_grad()
            # 计算分类损失，mmd损失和差异损失
            cls_loss, domain_loss, l1_loss,pl1_loss, cls_loss_total, features_source7, pred7, we7,outputs_target7, data_src7 = model(
                source_data7,
                target_data,
                source_label7,
                s_domain_label7,
                t_domain_label, mem_fea1,mem_fea2,mem_fea3,mem_fea4,mem_fea5,mem_fea6,mem_fea7,
                mem_cls1,mem_cls2,mem_cls3,mem_cls4,mem_cls5,mem_cls6,mem_cls7,idx,max_iter=num_iter,
                mark=7)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # features_target7 = features_target7 / torch.norm(features_target7, p=2, dim=1, keepdim=True)
            # dis = torch.mm(features_target7.detach(), mem_fea7.t())
            # _, pred7 = torch.max(dis, dim=1)

            # classifier_loss = nn.CrossEntropyLoss()(outputs_target, pred)
            # total_loss += args.tar_par * eff * classifier_loss
            # outputs_target7 = F.log_softmax(outputs_target7, dim=1)
            # classifier_loss7 = F.nll_loss(outputs_target7, pred7)


            #没有实例权重用下面两句
            # outputs_target7 = F.log_softmax(outputs_target7, dim=1)
            # classifier_loss7 = F.nll_loss(outputs_target7, pred7)
            #用到实例的权重
            outputs_target7 = F.log_softmax(outputs_target7, dim=1)
            classifier_loss7 = F.nll_loss(outputs_target7, pred7)
            classifier_loss7=torch.sum(we7 * classifier_loss7) / (torch.sum(we7).item())
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            gamma = 2 / (1 + math.exp(-10 * (i) / num_iter)) - 1
            ga = (i / num_iter) * 10


            # 参数敏感性6  2:x时
            loss = (cls_loss + gamma*(1*(domain_loss  )+0.5* (classifier_loss7 + cls_loss_total)))+ l1_loss

            loss.backward()
            optimizer.step()
            print('Train source7 iter: {}\tLoss: {:.6f}'.format(i, loss.item()))
            model.eval()
            with torch.no_grad():
                pred1, pred2, pred3, pred4, pred5, pred6, pred7, features_target1, features_target2, features_target3, features_target4, features_target5, features_target6, features_target7 = model(
                    target_data, mark=0)
                features_target1 = features_target1 / torch.norm(features_target1, p=2, dim=1, keepdim=True)
                pred1 = torch.nn.functional.log_softmax(pred1, dim=1)
                features_target2 = features_target2 / torch.norm(features_target2, p=2, dim=1, keepdim=True)
                pred2 = torch.nn.functional.log_softmax(pred2, dim=1)
                features_target3 = features_target3 / torch.norm(features_target3, p=2, dim=1, keepdim=True)
                pred3 = torch.nn.functional.log_softmax(pred3, dim=1)
                features_target4 = features_target4 / torch.norm(features_target4, p=2, dim=1, keepdim=True)
                pred4 = torch.nn.functional.log_softmax(pred4, dim=1)
                features_target5 = features_target5 / torch.norm(features_target5, p=2, dim=1, keepdim=True)
                pred5 = torch.nn.functional.log_softmax(pred5, dim=1)
                features_target6 = features_target6 / torch.norm(features_target6, p=2, dim=1, keepdim=True)
                pred6 = torch.nn.functional.log_softmax(pred6, dim=1)
                features_target7 = features_target7 / torch.norm(features_target7, p=2, dim=1, keepdim=True)
                pred7 = torch.nn.functional.log_softmax(pred7, dim=1)

            mem_fea1[idx] = 0.1 * mem_fea1[idx] + 0.9* features_target1.clone()
            mem_cls1[idx] = 0.1 * mem_cls1[idx] + 0.9 * pred1.clone()
            mem_fea2[idx] = 0.1 * mem_fea2[idx] + 0.9 * features_target2.clone()
            mem_cls2[idx] = 0.1 * mem_cls2[idx] + 0.9 * pred2.clone()
            mem_fea3[idx] = 0.1 * mem_fea3[idx] + 0.9 * features_target3.clone()
            mem_cls3[idx] = 0.1 * mem_cls3[idx] + 0.9 * pred3.clone()
            mem_fea4[idx] = 0.1 * mem_fea4[idx] + 0.9 * features_target4.clone()
            mem_cls4[idx] = 0.1 * mem_cls4[idx] + 0.9 * pred4.clone()
            mem_fea5[idx] = 0.1 * mem_fea5[idx] + 0.9 * features_target5.clone()
            mem_cls5[idx] = 0.1 * mem_cls5[idx] + 0.9 * pred5.clone()
            mem_fea6[idx] = 0.1 * mem_fea6[idx] + 0.9 * features_target6.clone()
            mem_cls6[idx] = 0.1 * mem_cls6[idx] + 0.9 * pred6.clone()
            mem_fea7[idx] = 0.1 * mem_fea7[idx] + 0.9 * features_target7.clone()
            mem_cls7[idx] = 0.1 * mem_cls7[idx] + 0.9 * pred7.clone()
            if i % len_dataloader == 0:
                t_qwk, t_test, acc = test(target_train_loader, model)
                qwkfilev.write(str(t_qwk))
                qwkfilev.write("\n")
                print('qwk of the  dataset: %f\n' % (t_qwk))
                if t_qwk > best_qwk_t:
                    best_qwk_t = t_qwk
                    torch.save(model, 'images/modelnew/model5.pth')
                    tab = 0
                else:
                    tab = tab + 1
                if tab ==10:
                    # 使用一个称为“tab”的参数来指定连续验证集损失没有提高的最大次数。如果在patience个epoch内验证集准确率没有提高，则停止训练并返回
                    print('Early stopping at epoch {}...'.format(i + 1))
                    break
                print('best_qwk_t of the dataset: %f\n' % (best_qwk_t))
            model.train()
        print('============ Summary ============= \n')
        qwkfilev.close()
        print('the epoch {} best_qwk_t of the dataset: {}\n'.format (epoch+1,best_qwk_t))
        print('============ Summary End============= \n')

    print('best qwk is ', best_qwk_t)
    print('============ Summary ============= \n')
    my_net = torch.load('images/modelnew/model5.pth')
    qwk_t,_,_ = test(test_loader, my_net)
    print('qwk of the %s dataset: %f\n' % ('test', qwk_t))
    print('============ Summary END============= \n')

def test(loader,model):
    model.eval()
    test_loss=0
    correct = 0
    num=0
    qwk,qwk1,qwk2,qwk3,qwk4,qwk5,qwk6,qwk7=0,0,0,0,0,0,0,0
    with torch.no_grad():
        for data, target,domain,idx in loader:
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.long()
                target = torch.flatten(target)
                target = target.to(device)
            data, target = Variable(data), Variable(target)
            pred1,pred2,pred3,pred4,pred5,pred6,pred7,features_target1,features_target2,features_target3,features_target4,features_target5,features_target6,features_target7 = model(data, mark=0)
            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)
            pred4 = torch.nn.functional.softmax(pred4, dim=1)
            pred5 = torch.nn.functional.softmax(pred5, dim=1)
            pred6 = torch.nn.functional.softmax(pred6, dim=1)
            pred7 = torch.nn.functional.softmax(pred7, dim=1)
            all_pred = (pred1 + pred2+pred3+pred4+pred5+pred6+pred7) / 7
            test_loss += F.nll_loss(F.log_softmax(all_pred, dim=1), target).item()
            num=num+1
            pred = all_pred.data.max(1)[1]
            pred1 = pred1.data.max(1)[1]
            pred2= pred2.data.max(1)[1]
            pred3 = pred3.data.max(1)[1]
            pred4 = pred4.data.max(1)[1]
            pred5 = pred5.data.max(1)[1]
            pred6 = pred6.data.max(1)[1]
            pred7 = pred7.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            predictions = np.array(pred.cpu())
            predictions1 = np.array(pred1.cpu())
            predictions2 = np.array(pred2.cpu())
            predictions3 = np.array(pred3.cpu())
            predictions4 = np.array(pred4.cpu())
            predictions5 = np.array(pred5.cpu())
            predictions6 = np.array(pred6.cpu())
            predictions7 = np.array(pred7.cpu())
            qwk+= quadratic_weighted_kappa(predictions, target.cpu())
            qwk1 += quadratic_weighted_kappa(predictions1, target.cpu())
            qwk2 += quadratic_weighted_kappa(predictions2, target.cpu())
            qwk3 += quadratic_weighted_kappa(predictions3, target.cpu())
            qwk4 += quadratic_weighted_kappa(predictions4, target.cpu())
            qwk5 += quadratic_weighted_kappa(predictions5, target.cpu())
            qwk6 += quadratic_weighted_kappa(predictions6, target.cpu())
            qwk7 += quadratic_weighted_kappa(predictions7, target.cpu())
        print( '\nsource1 qwk1 {}, source2 qwk2 {},source3 qwk3 {}, source4 qwk4 {},source5 qwk5 {}, source6 qwk6 {},source7 qwk7 {}'.format(
                qwk1/num, qwk2/num, qwk3/num, qwk4/num, qwk5/num, qwk6/num, qwk7/num))
        acc=(100. * correct)/len(loader.dataset)
        print("correct:{}".format(acc))
        testloss=test_loss/len(loader.dataset)
        qwk=qwk/num
        print('\ntest_loss {}'.format(testloss))
    return qwk,testloss,acc

if __name__ == '__main__':
    train()

