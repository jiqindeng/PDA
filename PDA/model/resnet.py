import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from metrics import quadratic_weighted_kappa
from options import args
from functions import ReverseLayerF
from sent_att_model import SentAttNet
# from sent_att_modelone import SentAttNet
from word_att_model import WordAttNet
device = torch.device("cuda:0")
class ADDneck(nn.Module):

    def __init__(self, sent_hidden_size):
        super(ADDneck, self).__init__()
        self.lstm=nn.LSTM(sent_hidden_size,sent_hidden_size)
        self.fc1=nn.Linear(sent_hidden_size,sent_hidden_size)
        self.bn1=nn.BatchNorm1d(sent_hidden_size)
        self.relu1=nn.ReLU(True)
        self.droput1=nn.Dropout()
        self.fc2 = nn.Linear(sent_hidden_size, sent_hidden_size)
    def forward(self, x):
        self.lstm.flatten_parameters()
        out,hidden=self.lstm(x)
        out=out.squeeze(0)
        out=self.fc1(out)
        out=self.bn1(out)
        out=self.relu1(out)
        out=self.droput1(out)
        out=self.fc2(out)
        return out

class ADDneckLinear(nn.Module):

    def __init__(self, sent_hidden_size):
        super(ADDneckLinear, self).__init__()
        self.fc1=nn.Linear(sent_hidden_size,sent_hidden_size)
        self.relu1=nn.ReLU(True)
    def forward(self, x):
        out=x.squeeze(0)
        out=self.fc1(out)
        out=self.relu1(out)
        return out
class HierNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, embed_table,
                 max_sent_length, max_word_length):
        super(HierNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(embed_table,word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)
        self._init_hidden_state()
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.to(device)
            self.sent_hidden_state = self.sent_hidden_state.to(device)
    def forward(self, input):
        # print("input.size():{}".format(input.size()))---torch.Size([5, 97, 50])
        #进入的是batchsize=5，其中max sentence num = 97，max sentence length = 50
        #也就是说进入5篇文章，每篇文章包含97个句子，每个句子包含50个单词
        output_list = torch.empty(0,).to(device)
        input = input.permute(1, 0, 2)
        # print("input.size():{}".format(input.size()))---input.size():torch.Size([97, 5, 50])
        # print("进入循环：")
        for i in input:
            output = self.word_att_net(i)
            # print("退出word_att_net")
            output_list = torch.cat((output_list,output))
            # print(output_list.shape)
            #第一次循环结束是torch.Size([1, 5, 100])，最后一次循环是torch.Size([97, 5, 100])
        # print("word_att_net结束，开始sent_att_net")
        output= self.sent_att_net(output_list)
        # print(output.shape)---torch.Size([1, 5, 100])
        #----需要处理的数据：5篇文章，每篇文章有1篇作文，每个作文用100个句子表示
        return output
def calc_coeff(iter_num,max_iter,num_epochs):
    p = float(iter_num + max_iter) / num_epochs / (max_iter/num_epochs)
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    return alpha
def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
class domain_classifier(nn.Module):

    def __init__(self, sent_hidden_size):
        super(domain_classifier, self).__init__()
        self.fc1=nn.Linear(sent_hidden_size,sent_hidden_size)
        self.bn1=nn.BatchNorm1d(sent_hidden_size)
        self.relu1=nn.ReLU(True)
        self.droput = nn.Dropout()
        self.fc2 = nn.Linear(sent_hidden_size, sent_hidden_size)
        self.bn2 = nn.BatchNorm1d(sent_hidden_size)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(sent_hidden_size, 1)
        self.iter_num = 0
    def forward(self, x,max_iter,i):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num,max_iter,args.num_epochs)
        reverse_feature = ReverseLayerF.apply(x, coeff)
        out=self.fc1(reverse_feature)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
class domain_classifier1(nn.Module):

    def __init__(self, sent_hidden_size):
        super(domain_classifier1, self).__init__()
        self.fc1=nn.Linear(sent_hidden_size,sent_hidden_size)
        self.bn1=nn.BatchNorm1d(sent_hidden_size)
        self.relu1=nn.ReLU(True)
        self.droput = nn.Dropout()
        self.fc2 = nn.Linear(sent_hidden_size, sent_hidden_size)
        self.bn2 = nn.BatchNorm1d(sent_hidden_size)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(sent_hidden_size, 1)
        self.iter_num = 0
        self.num=0
    def forward(self, x,max_iter):
        if self.training:
            if self.num%args.num_epochs==0:
                self.iter_num += 1
            self.num+=1
        coeff = calc_coeff(self.iter_num,max_iter,args.num_epochs)
        reverse_feature = ReverseLayerF.apply(x, coeff)
        out=self.fc1(reverse_feature)
        # out=self.bn1(out)
        out=self.relu1(out)
        out = self.fc3(out)
        return out
class MFSAN(nn.Module):

    def __init__(self,batch_size,embed_table,max_sentnum,max_sentlen):
        super(MFSAN, self).__init__()
        #将共享层改成了CNN-ATT的形式
        self.sharedNet = HierNet(100, 100, batch_size, embed_table, max_sentnum, max_sentlen)
        # LSTM
        self.sonnet1 = ADDneck(100)
        self.sonnet2 = ADDneck(100)
        self.sonnet3 = ADDneck(100)
        self.sonnet4 = ADDneck(100)
        self.sonnet5 = ADDneck(100)
        self.sonnet6 = ADDneck(100)
        self.sonnet7 = ADDneck(100)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.cls_cls10 = domain_classifier(100)
        self.cls_cls11= domain_classifier(100)
        self.cls_cls12 = domain_classifier(100)
        self.cls_cls13 = domain_classifier(100)

        self.cls_cls20 = domain_classifier(100)
        self.cls_cls21 = domain_classifier(100)
        self.cls_cls22 = domain_classifier(100)
        self.cls_cls23 = domain_classifier(100)

        self.cls_cls30 = domain_classifier(100)
        self.cls_cls31 = domain_classifier(100)
        self.cls_cls32 = domain_classifier(100)
        self.cls_cls33 = domain_classifier(100)

        self.cls_cls40 = domain_classifier(100)
        self.cls_cls41 = domain_classifier(100)
        self.cls_cls42 = domain_classifier(100)
        self.cls_cls43 = domain_classifier(100)

        self.cls_cls50 = domain_classifier(100)
        self.cls_cls51 = domain_classifier(100)
        self.cls_cls52 = domain_classifier(100)
        self.cls_cls53 = domain_classifier(100)

        self.cls_cls60 = domain_classifier(100)
        self.cls_cls61 = domain_classifier(100)
        self.cls_cls62 = domain_classifier(100)
        self.cls_cls63 = domain_classifier(100)

        self.cls_cls70 = domain_classifier(100)
        self.cls_cls71 = domain_classifier(100)
        self.cls_cls72 = domain_classifier(100)
        self.cls_cls73 = domain_classifier(100)
 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        self.cls_fc_son1 =classification(100)
        self.cls_fc_son2 = classification(100)
        self.cls_fc_son3 = classification(100)
        self.cls_fc_son4 = classification(100)
        self.cls_fc_son5 = classification(100)
        self.cls_fc_son6 = classification(100)
        self.cls_fc_son7 = classification(100)

        self.domain_cls1 = domain_classifier(100)
        self.domain_cls2 = domain_classifier(100)
        self.domain_cls3 = domain_classifier(100)
        self.domain_cls4 = domain_classifier(100)
        self.domain_cls5 = domain_classifier(100)
        self.domain_cls6 = domain_classifier(100)
        self.domain_cls7 = domain_classifier(100)

    def forward(self, data_src, data_tgt=0, label_src=0, s_domain_label=0, t_domain_label=0, mem_fea1=0, mem_fea2=0,mem_fea3=0,mem_fea4=0,mem_fea5=0,mem_fea6=0,mem_fea7=0,mem_cls1=0,mem_cls2=0,mem_cls3=0,mem_cls4=0,mem_cls5=0,mem_cls6=0,mem_cls7=0,idx=0,max_iter=0,mark=1):
        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            data_tgt1 = self.sonnet1(data_tgt)
            data_tgt_son1 = self.domain_cls1(data_tgt1, max_iter,1)
            pred_tgt_son1 = self.cls_fc_son1(data_tgt1)

            data_tgt2 = self.sonnet2(data_tgt)
            data_tgt_son2 = self.domain_cls2(data_tgt2, max_iter,1)
            pred_tgt_son2 = self.cls_fc_son2(data_tgt2)

            data_tgt3 = self.sonnet3(data_tgt)
            data_tgt_son3 = self.domain_cls3(data_tgt3, max_iter,1)
            # data_tgt_son3 = data_tgt_son3.view(data_tgt_son3.size(0), -1)
            pred_tgt_son3 = self.cls_fc_son3(data_tgt3)

            data_tgt4 = self.sonnet4(data_tgt)
            data_tgt_son4 = self.domain_cls4(data_tgt4, max_iter,1)
            # data_tgt_son4 = data_tgt_son4.view(data_tgt_son4.size(0), -1)
            pred_tgt_son4 = self.cls_fc_son4(data_tgt4)

            data_tgt5 = self.sonnet5(data_tgt)
            data_tgt_son5 = self.domain_cls5(data_tgt5, max_iter,1)
            # data_tgt_son5 = data_tgt_son5.view(data_tgt_son5.size(0), -1)
            pred_tgt_son5 = self.cls_fc_son5(data_tgt5)

            data_tgt6 = self.sonnet6(data_tgt)
            data_tgt_son6 = self.domain_cls6(data_tgt6, max_iter,1)
            # data_tgt_son6 = data_tgt_son6.view(data_tgt_son6.size(0), -1)
            pred_tgt_son6 = self.cls_fc_son6(data_tgt6)

            data_tgt7 = self.sonnet7(data_tgt)
            data_tgt_son7 = self.domain_cls7(data_tgt7, max_iter,1)
            # data_tgt_son7 = data_tgt_son7.view(data_tgt_son7.size(0), -1)
            pred_tgt_son7 = self.cls_fc_son7(data_tgt7)

            # &&&&&&&&&&&&&&&&&&&&&&&&&下面是打伪标签用的&&&&&&&&&&&&&&&&&&&&&&&&&
            # features_target1 = data_tgt1 / torch.norm(data_tgt1, p=2, dim=1, keepdim=True)
            # dis1 = torch.mm(features_target1.detach(), mem_fea1.t())
            # _, pred1 = torch.max(dis1, dim=1)
            # features_target2 = data_tgt2 / torch.norm(data_tgt2, p=2, dim=1, keepdim=True)
            # dis2 = torch.mm(features_target2.detach(), mem_fea2.t())
            # _, pred2 = torch.max(dis2, dim=1)
            # features_target3 = data_tgt3 / torch.norm(data_tgt3, p=2, dim=1, keepdim=True)
            # dis3 = torch.mm(features_target3.detach(), mem_fea3.t())
            # _, pred3 = torch.max(dis3, dim=1)
            # features_target4 = data_tgt4 / torch.norm(data_tgt4, p=2, dim=1, keepdim=True)
            # dis4 = torch.mm(features_target4.detach(), mem_fea4.t())
            # _, pred4 = torch.max(dis4, dim=1)
            # features_target5 = data_tgt5 / torch.norm(data_tgt5, p=2, dim=1, keepdim=True)
            # dis5 = torch.mm(features_target5.detach(), mem_fea5.t())
            # _, pred5= torch.max(dis5, dim=1)
            # features_target6 = data_tgt6 / torch.norm(data_tgt6, p=2, dim=1, keepdim=True)
            # dis6 = torch.mm(features_target6.detach(), mem_fea6.t())
            # _, pred6 = torch.max(dis6, dim=1)
            #
            # features_target7 = data_tgt7 / torch.norm(data_tgt7, p=2, dim=1, keepdim=True)
            # dis7 = torch.mm(features_target7.detach(), mem_fea7.t())
            # _, pred7 = torch.max(dis7, dim=1)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            dis1 = -torch.mm(data_tgt1.detach(), mem_fea1.t())
            for di in range(dis1.size(0)):
                dis1[di, idx[di]] = torch.max(dis1)
            _, p11 = torch.sort(dis1, dim=1)
            w = torch.zeros(data_tgt1.size(0), mem_fea1.size(0)).to(device)
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p11[wi, wj]] = 1 / 5
            weight1_, pred1 = torch.max(w.mm(mem_cls1), 1)
            dis2 = -torch.mm(data_tgt2.detach(), mem_fea2.t())
            for di in range(dis2.size(0)):
                dis2[di, idx[di]] = torch.max(dis2)
            _, p12 = torch.sort(dis2, dim=1)
            w = torch.zeros(data_tgt2.size(0), mem_fea2.size(0)).to(device)
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p12[wi, wj]] = 1 / 5
            weight2_, pred2 = torch.max(w.mm(mem_cls2), 1)

            dis3 = -torch.mm(data_tgt3.detach(), mem_fea3.t())
            for di in range(dis3.size(0)):
                dis3[di, idx[di]] = torch.max(dis3)
            _, p13 = torch.sort(dis3, dim=1)
            w = torch.zeros(data_tgt3.size(0), mem_fea3.size(0)).to(device)
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p13[wi, wj]] = 1 / 5
            weight3_, pred3 = torch.max(w.mm(mem_cls3), 1)

            dis4 = -torch.mm(data_tgt4.detach(), mem_fea4.t())
            for di in range(dis4.size(0)):
                dis4[di, idx[di]] = torch.max(dis4)
            _, p14 = torch.sort(dis4, dim=1)
            w = torch.zeros(data_tgt4.size(0), mem_fea4.size(0)).to(device)
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p14[wi, wj]] = 1 / 5
            weight4_, pred4 = torch.max(w.mm(mem_cls4), 1)

            dis5 = -torch.mm(data_tgt5.detach(), mem_fea5.t())
            for di in range(dis5.size(0)):
                dis5[di, idx[di]] = torch.max(dis5)
            _, p15 = torch.sort(dis5, dim=1)
            w = torch.zeros(data_tgt5.size(0), mem_fea5.size(0)).to(device)
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p15[wi, wj]] = 1 / 5
            weight5_, pred5 = torch.max(w.mm(mem_cls5), 1)

            dis6 = -torch.mm(data_tgt6.detach(), mem_fea6.t())
            for di in range(dis6.size(0)):
                dis6[di, idx[di]] = torch.max(dis6)
            _, p16 = torch.sort(dis6, dim=1)
            w = torch.zeros(data_tgt6.size(0), mem_fea6.size(0)).to(device)
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p16[wi, wj]] = 1 / 5
            weight6_, pred6 = torch.max(w.mm(mem_cls6), 1)

            dis7 = -torch.mm(data_tgt7.detach(), mem_fea7.t())
            for di in range(dis7.size(0)):
                dis7[di, idx[di]] = torch.max(dis7)
            _, p17 = torch.sort(dis7, dim=1)
            w = torch.zeros(data_tgt7.size(0), mem_fea7.size(0)).to(device)
            for wi in range(w.size(0)):
                for wj in range(5):
                    w[wi][p17[wi, wj]] = 1 / 5
            weight7_, pred7 = torch.max(w.mm(mem_cls7), 1)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            if mark == 1:
                data_src1 = self.sonnet1(data_src)
                domain_pred_src = self.domain_cls1(data_src1, max_iter,1)
                domain_src_loss = nn.BCELoss()(torch.sigmoid(domain_pred_src), s_domain_label)
                domain_tgt_loss = nn.BCELoss()(torch.sigmoid(data_tgt_son1), t_domain_label)
                domain_loss = (domain_src_loss + domain_tgt_loss)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son1, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son3, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son4, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son5, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son6, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son7, dim=1)))
                pred_src = self.cls_fc_son1(data_src1)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                #￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
                pl1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(dis1, dim=1)
                                               - torch.nn.functional.softmax(dis2, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis1, dim=1)
                                                - torch.nn.functional.softmax(dis3, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis1, dim=1)
                                                - torch.nn.functional.softmax(dis4, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis1, dim=1)
                                                - torch.nn.functional.softmax(dis5, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis1, dim=1)
                                                - torch.nn.functional.softmax(dis6, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis1, dim=1)
                                                - torch.nn.functional.softmax(dis7, dim=1)))
                #……………………………………………………………………………………………………
                j, m = 0, 0
                num0, num1, num2, num3 = 0, 0, 0, 0
                for label in label_src:
                    label=label.unsqueeze(0)
                    all_data_src1=data_src1.data[j].unsqueeze(0)
                    if label.data == 0:
                        num0=num0+1
                        if num0==1:
                            all_data_source0=all_data_src1
                            all_label_source0=label
                        else:
                            all_data_source0 = torch.cat([all_data_source0,all_data_src1],dim=0)
                            all_label_source0=torch.cat([all_label_source0,label],dim=0)
                        # source_fea = self.cls_cls10(data_src1.data[j].to(device).to(device), max_iter)
                    elif label.data == 1:
                        num1=num1+1
                        if num1== 1:
                            all_data_source1 = all_data_src1
                            all_label_source1 = label
                        else:
                            all_data_source1 = torch.cat([all_data_source1, all_data_src1],dim=0)
                            all_label_source1 = torch.cat([all_label_source1, label], dim=0)
                        # source_fea = self.cls_cls11(data_src1.data[j].to(device), max_iter)
                    elif label.data == 2:
                        num2=num2+1
                        if num2== 1:
                            all_data_source2 = all_data_src1
                            all_label_source2 = label
                        else:
                            all_data_source2 = torch.cat([all_data_source2, all_data_src1],dim=0)
                            all_label_source2 = torch.cat([all_label_source2, label], dim=0)
                        # source_fea = self.cls_cls12(data_src1.data[j].to(device), max_iter)
                    elif label.data == 3:
                        num3=num3+1
                        if num3== 1:
                            all_data_source3 = all_data_src1
                            all_label_source3 = label
                        else:
                            all_data_source3 = torch.cat([all_data_source3, all_data_src1],dim=0)
                            all_label_source3 = torch.cat([all_label_source3, label], dim=0)
                        # source_fea = self.cls_cls13(data_src1.data[j].to(device), max_iter)
                    j = j + 1
                all_domain_src_loss0,all_domain_src_loss1,all_domain_src_loss3,all_domain_src_loss2=0,0,0,0
                if num0!=0:
                    s_label0 = Variable(torch.zeros(num0,1).to(device))
                    all_domain_pred_src0 = self.cls_cls10(all_data_source0, max_iter,0)
                    all_domain_src_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src0), s_label0)
                if num1!=0:
                    s_label1 = Variable(torch.zeros(num1,1).to(device))
                    all_domain_pred_src1 = self.cls_cls11(all_data_source1, max_iter,0)
                    all_domain_src_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src1), s_label1)
                if num2!=0:
                    s_label2 = Variable(torch.zeros(num2,1).to(device))
                    all_domain_pred_src2 = self.cls_cls12(all_data_source2, max_iter,0)
                    all_domain_src_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src2), s_label2)
                if num3!=0:
                    s_label3 = Variable(torch.zeros(num3,1).to(device))
                    all_domain_pred_src3 = self.cls_cls13(all_data_source3, max_iter,0)
                    all_domain_src_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src3), s_label3)
                all_domain_src_loss=all_domain_src_loss0+all_domain_src_loss1+all_domain_src_loss3+all_domain_src_loss2


                tnum0,tnum1,tnum2,tnum3=0,0,0,0
                for label in pred1:
                    label = label.unsqueeze(0)
                    all_data_tgt1 = data_tgt1.data[m].unsqueeze(0)
                    if label.data == 0:
                        tnum0 = tnum0 + 1
                        if tnum0 == 1:
                            all_data_target0 = all_data_tgt1
                            all_label_target0 = label
                        else:
                            all_data_target0 = torch.cat([all_data_target0, all_data_tgt1], dim=0)
                            all_label_target0 = torch.cat([all_label_target0, label], dim=0)

                    elif label.data == 1:
                        tnum1 = tnum1 + 1
                        if tnum1 == 1:
                            all_data_target1 = all_data_tgt1
                            all_label_target1 = label
                        else:
                            all_data_target1 = torch.cat([all_data_target1, all_data_tgt1], dim=0)
                            all_label_target1 = torch.cat([all_label_target1, label], dim=0)
                    elif label.data == 2:
                        tnum2 = tnum2 + 1
                        if tnum2 == 1:
                            all_data_target2 = all_data_tgt1
                            all_label_target2 = label
                        else:
                            all_data_target2 = torch.cat([all_data_target2, all_data_tgt1], dim=0)
                            all_label_target2 = torch.cat([all_label_target2, label], dim=0)
                    elif label.data == 3:
                        tnum3 = tnum3 + 1
                        if tnum3 == 1:
                            all_data_target3 = all_data_tgt1
                            all_label_target3 = label
                        else:
                            all_data_target3 = torch.cat([all_data_target3, all_data_tgt1], dim=0)
                            all_label_target3 = torch.cat([all_label_target3, label], dim=0)
                    m = m + 1
                all_domain_t_loss0 ,all_domain_t_loss1 , all_domain_t_loss3 ,all_domain_t_loss2=0,0,0,0
                if tnum0!=0:
                    t_label0 = Variable(torch.zeros(tnum0,1).to(device))
                    all_domain_t_src0 = self.cls_cls10(all_data_target0, max_iter,0)
                    all_domain_t_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_t_src0), t_label0)
                if tnum1!=0:
                    t_label1 = Variable(torch.zeros(tnum1,1).to(device))
                    all_domain_t_src1 = self.cls_cls11(all_data_target1, max_iter,0)
                    all_domain_t_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_t_src1), t_label1)
                if tnum2!=0:
                    t_label2 = Variable(torch.zeros(tnum2,1).to(device))
                    all_domain_t_src2 = self.cls_cls12(all_data_target2, max_iter,0)
                    all_domain_t_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_t_src2), t_label2)
                if tnum3!=0:
                    t_label3 = Variable(torch.zeros(tnum3,1).to(device))
                    all_domain_t_src3 = self.cls_cls13(all_data_target3, max_iter,0)
                    all_domain_t_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_t_src3), t_label3)
                all_domain_t_loss = all_domain_t_loss0 + all_domain_t_loss1 + all_domain_t_loss3 + all_domain_t_loss2

                cls_loss_total = all_domain_src_loss + all_domain_t_loss
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                return cls_loss, domain_loss, l1_loss / 6,pl1_loss/6, cls_loss_total, data_src1, pred1,weight1_, pred_tgt_son1,data_src1

            if mark == 2:
                data_src2 = self.sonnet2(data_src)
                domain_pred_src = self.domain_cls2(data_src2, max_iter,1)
                # domain_src_loss = F.nll_loss(F.log_softmax(domain_pred_src, dim=1), s_domain_label)
                # domain_tgt_loss = F.nll_loss(F.log_softmax(data_tgt_son2, dim=1), t_domain_label)
                domain_src_loss = nn.BCELoss()(torch.sigmoid(domain_pred_src), s_domain_label)
                domain_tgt_loss = nn.BCELoss()(torch.sigmoid(data_tgt_son2), t_domain_label)
                domain_loss = domain_src_loss + domain_tgt_loss
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son2, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son3, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son4, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son5, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son6, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son7, dim=1)))
                pred_src = self.cls_fc_son2(data_src2)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pl1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(dis2, dim=1)
                                                - torch.nn.functional.softmax(dis1, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis2, dim=1)
                                                 - torch.nn.functional.softmax(dis3, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis2, dim=1)
                                                 - torch.nn.functional.softmax(dis4, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis2, dim=1)
                                                 - torch.nn.functional.softmax(dis5, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis2, dim=1)
                                                 - torch.nn.functional.softmax(dis6, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis2, dim=1)
                                                 - torch.nn.functional.softmax(dis7, dim=1)))
                # ……………………………………………………………………………………………………
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # &&&&&&&&&&&&&&&&&&&&&&&1,2,3,4,5,6,7个源域的某个类看做一个大源域&&&&&&&&&&&&&&&&&&&&&&&&&
                j, m = 0, 0
                num0, num1, num2, num3 = 0, 0, 0, 0
                for label in label_src:
                    label = label.unsqueeze(0)
                    all_data_src2 = data_src2.data[j].unsqueeze(0)
                    if label.data == 0:
                        num0 = num0 + 1
                        if num0 == 1:
                            all_data_source0 = all_data_src2
                            all_label_source0 = label
                        else:
                            all_data_source0 = torch.cat([all_data_source0, all_data_src2], dim=0)
                            all_label_source0 = torch.cat([all_label_source0, label], dim=0)
                        # source_fea = self.cls_cls10(data_src1.data[j].to(device).to(device), max_iter)
                    elif label.data == 1:
                        num1 = num1 + 1
                        if num1 == 1:
                            all_data_source1 = all_data_src2
                            all_label_source1 = label
                        else:
                            all_data_source1 = torch.cat([all_data_source1, all_data_src2], dim=0)
                            all_label_source1 = torch.cat([all_label_source1, label], dim=0)
                        # source_fea = self.cls_cls11(data_src1.data[j].to(device), max_iter)
                    elif label.data == 2:
                        num2 = num2 + 1
                        # print(label.data)
                        # tensor([2], device='cuda:0')
                        # print(label)
                        # tensor([2], device='cuda:0')
                        # print(all_label_source0.size())
                        # torch.Size([1])
                        if num2 == 1:
                            all_data_source2 = all_data_src2
                            all_label_source2 = label
                            # print(all_label_source1)
                            # tensor([2], device='cuda:0')
                        else:
                            all_data_source2 = torch.cat([all_data_source2, all_data_src2], dim=0)
                            all_label_source2 = torch.cat([all_label_source2, label], dim=0)
                        # source_fea = self.cls_cls12(data_src1.data[j].to(device), max_iter)
                    elif label.data == 3:
                        num3 = num3 + 1
                        if num3 == 1:
                            all_data_source3 = all_data_src2
                            all_label_source3 = label
                        else:
                            all_data_source3 = torch.cat([all_data_source3, all_data_src2], dim=0)
                            all_label_source3 = torch.cat([all_label_source3, label], dim=0)
                        # source_fea = self.cls_cls13(data_src1.data[j].to(device), max_iter)
                    j = j + 1
                all_domain_src_loss0, all_domain_src_loss1, all_domain_src_loss3, all_domain_src_loss2 = 0, 0, 0, 0
                if num0 != 0:
                    s_label0 = Variable(torch.zeros(num0, 1).to(device))
                    all_domain_pred_src0 = self.cls_cls10(all_data_source0, max_iter, 0)
                    all_domain_src_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src0), s_label0)
                if num1 != 0:
                    s_label1 = Variable(torch.zeros(num1, 1).to(device))
                    all_domain_pred_src1 = self.cls_cls11(all_data_source1, max_iter, 0)
                    all_domain_src_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src1), s_label1)
                if num2 != 0:
                    s_label2 = Variable(torch.zeros(num2, 1).to(device))
                    all_domain_pred_src2 = self.cls_cls12(all_data_source2, max_iter, 0)
                    all_domain_src_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src2), s_label2)
                if num3 != 0:
                    s_label3 = Variable(torch.zeros(num3, 1).to(device))
                    all_domain_pred_src3 = self.cls_cls13(all_data_source3, max_iter, 0)
                    all_domain_src_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src3), s_label3)
                all_domain_src_loss = all_domain_src_loss0 + all_domain_src_loss1 + all_domain_src_loss3 + all_domain_src_loss2

                tnum0, tnum1, tnum2, tnum3 = 0, 0, 0, 0
                for label in pred2:
                    label = label.unsqueeze(0)
                    all_data_tgt2 = data_tgt2.data[m].unsqueeze(0)
                    if label.data == 0:
                        tnum0 = tnum0 + 1
                        if tnum0 == 1:
                            all_data_target0 = all_data_tgt2
                            all_label_target0 = label
                        else:
                            all_data_target0 = torch.cat([all_data_target0, all_data_tgt2], dim=0)
                            all_label_target0 = torch.cat([all_label_target0, label], dim=0)

                    elif label.data == 1:
                        tnum1 = tnum1 + 1
                        if tnum1 == 1:
                            all_data_target1 = all_data_tgt2
                            all_label_target1 = label
                        else:
                            all_data_target1 = torch.cat([all_data_target1, all_data_tgt2], dim=0)
                            all_label_target1 = torch.cat([all_label_target1, label], dim=0)
                    elif label.data == 2:
                        tnum2 = tnum2 + 1
                        if tnum2 == 1:
                            all_data_target2 = all_data_tgt2
                            all_label_target2 = label
                        else:
                            all_data_target2 = torch.cat([all_data_target2, all_data_tgt2], dim=0)
                            all_label_target2 = torch.cat([all_label_target2, label], dim=0)
                    elif label.data == 3:
                        tnum3 = tnum3 + 1
                        if tnum3 == 1:
                            all_data_target3 = all_data_tgt2
                            all_label_target3 = label
                        else:
                            all_data_target3 = torch.cat([all_data_target3, all_data_tgt2], dim=0)
                            all_label_target3 = torch.cat([all_label_target3, label], dim=0)
                    m = m + 1
                all_domain_t_loss0, all_domain_t_loss1, all_domain_t_loss3, all_domain_t_loss2 = 0, 0, 0, 0
                if tnum0 != 0:
                    t_label0 = Variable(torch.zeros(tnum0, 1).to(device))
                    all_domain_t_src0 = self.cls_cls10(all_data_target0, max_iter, 0)
                    all_domain_t_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_t_src0), t_label0)
                if tnum1 != 0:
                    t_label1 = Variable(torch.zeros(tnum1, 1).to(device))
                    all_domain_t_src1 = self.cls_cls11(all_data_target1, max_iter, 0)
                    all_domain_t_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_t_src1), t_label1)
                if tnum2 != 0:
                    t_label2 = Variable(torch.zeros(tnum2, 1).to(device))
                    all_domain_t_src2 = self.cls_cls12(all_data_target2, max_iter, 0)
                    all_domain_t_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_t_src2), t_label2)
                if tnum3 != 0:
                    t_label3 = Variable(torch.zeros(tnum3, 1).to(device))
                    all_domain_t_src3 = self.cls_cls13(all_data_target3, max_iter, 0)
                    all_domain_t_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_t_src3), t_label3)
                all_domain_t_loss = all_domain_t_loss0 + all_domain_t_loss1 + all_domain_t_loss3 + all_domain_t_loss2
                cls_loss_total = all_domain_src_loss + all_domain_t_loss
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

                return cls_loss, domain_loss, l1_loss / 6,pl1_loss/6, cls_loss_total, data_src2, pred2,weight2_, pred_tgt_son2,data_src2
            if mark == 3:
                data_src3 = self.sonnet3(data_src)
                domain_pred_src = self.domain_cls3(data_src3, max_iter,1)
                # domain_src_loss = F.nll_loss(F.log_softmax(domain_pred_src, dim=1), s_domain_label)
                # domain_tgt_loss = F.nll_loss(F.log_softmax(data_tgt_son3, dim=1), t_domain_label)
                domain_src_loss = nn.BCELoss()(torch.sigmoid(domain_pred_src), s_domain_label)
                domain_tgt_loss = nn.BCELoss()(torch.sigmoid(data_tgt_son3), t_domain_label)
                domain_loss = domain_src_loss + domain_tgt_loss
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son3, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son4, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son5, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son6, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son7, dim=1)))
                pred_src = self.cls_fc_son3(data_src3)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pl1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(dis3, dim=1)
                                                - torch.nn.functional.softmax(dis1, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis3, dim=1)
                                                 - torch.nn.functional.softmax(dis2, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis3, dim=1)
                                                 - torch.nn.functional.softmax(dis4, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis3, dim=1)
                                                 - torch.nn.functional.softmax(dis5, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis3, dim=1)
                                                 - torch.nn.functional.softmax(dis6, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis3, dim=1)
                                                 - torch.nn.functional.softmax(dis7, dim=1)))
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # &&&&&&&&&&&&&&&&&&&&&&&1,2,3,4,5,6,7个源域的某个类看做一个大源域&&&&&&&&&&&&&&&&&&&&&&&&&
                j, m = 0, 0
                num0, num1, num2, num3 = 0, 0, 0, 0
                for label in label_src:
                    label = label.unsqueeze(0)
                    all_data_src3 = data_src3.data[j].unsqueeze(0)
                    if label.data == 0:
                        num0 = num0 + 1
                        if num0 == 1:
                            all_data_source0 = all_data_src3
                            all_label_source0 = label
                        else:
                            all_data_source0 = torch.cat([all_data_source0, all_data_src3], dim=0)
                            all_label_source0 = torch.cat([all_label_source0, label], dim=0)
                        # source_fea = self.cls_cls10(data_src1.data[j].to(device).to(device), max_iter)
                    elif label.data == 1:
                        num1 = num1 + 1
                        if num1 == 1:
                            all_data_source1 = all_data_src3
                            all_label_source1 = label
                        else:
                            all_data_source1 = torch.cat([all_data_source1, all_data_src3], dim=0)
                            all_label_source1 = torch.cat([all_label_source1, label], dim=0)
                        # source_fea = self.cls_cls11(data_src1.data[j].to(device), max_iter)
                    elif label.data == 2:
                        num2 = num2 + 1
                        # print(label.data)
                        # tensor([2], device='cuda:0')
                        # print(label)
                        # tensor([2], device='cuda:0')
                        # print(all_label_source0.size())
                        # torch.Size([1])
                        if num2 == 1:
                            all_data_source2 = all_data_src3
                            all_label_source2 = label
                            # print(all_label_source1)
                            # tensor([2], device='cuda:0')
                        else:
                            all_data_source2 = torch.cat([all_data_source2, all_data_src3], dim=0)
                            all_label_source2 = torch.cat([all_label_source2, label], dim=0)
                        # source_fea = self.cls_cls12(data_src1.data[j].to(device), max_iter)
                    elif label.data == 3:
                        num3 = num3 + 1
                        if num3 == 1:
                            all_data_source3 = all_data_src3
                            all_label_source3 = label
                        else:
                            all_data_source3 = torch.cat([all_data_source3, all_data_src3], dim=0)
                            all_label_source3 = torch.cat([all_label_source3, label], dim=0)
                        # source_fea = self.cls_cls13(data_src1.data[j].to(device), max_iter)
                    j = j + 1
                all_domain_src_loss0, all_domain_src_loss1, all_domain_src_loss3, all_domain_src_loss2 = 0, 0, 0, 0
                if num0 != 0:
                    s_label0 = Variable(torch.zeros(num0, 1).to(device))
                    all_domain_pred_src0 = self.cls_cls10(all_data_source0, max_iter, 0)
                    all_domain_src_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src0), s_label0)
                if num1 != 0:
                    s_label1 = Variable(torch.zeros(num1, 1).to(device))
                    all_domain_pred_src1 = self.cls_cls11(all_data_source1, max_iter, 0)
                    all_domain_src_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src1), s_label1)
                if num2 != 0:
                    s_label2 = Variable(torch.zeros(num2, 1).to(device))
                    all_domain_pred_src2 = self.cls_cls12(all_data_source2, max_iter, 0)
                    all_domain_src_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src2), s_label2)
                if num3 != 0:
                    s_label3 = Variable(torch.zeros(num3, 1).to(device))
                    all_domain_pred_src3 = self.cls_cls13(all_data_source3, max_iter, 0)
                    all_domain_src_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src3), s_label3)
                all_domain_src_loss = all_domain_src_loss0 + all_domain_src_loss1 + all_domain_src_loss3 + all_domain_src_loss2

                tnum0, tnum1, tnum2, tnum3 = 0, 0, 0, 0
                for label in pred3:
                    label = label.unsqueeze(0)
                    all_data_tgt3 = data_tgt3.data[m].unsqueeze(0)
                    if label.data == 0:
                        tnum0 = tnum0 + 1
                        if tnum0 == 1:
                            all_data_target0 = all_data_tgt3
                            all_label_target0 = label
                        else:
                            all_data_target0 = torch.cat([all_data_target0, all_data_tgt3], dim=0)
                            all_label_target0 = torch.cat([all_label_target0, label], dim=0)

                    elif label.data == 1:
                        tnum1 = tnum1 + 1
                        if tnum1 == 1:
                            all_data_target1 = all_data_tgt3
                            all_label_target1 = label
                        else:
                            all_data_target1 = torch.cat([all_data_target1, all_data_tgt3], dim=0)
                            all_label_target1 = torch.cat([all_label_target1, label], dim=0)
                    elif label.data == 2:
                        tnum2 = tnum2 + 1
                        if tnum2 == 1:
                            all_data_target2 = all_data_tgt3
                            all_label_target2 = label
                        else:
                            all_data_target2 = torch.cat([all_data_target2, all_data_tgt3], dim=0)
                            all_label_target2 = torch.cat([all_label_target2, label], dim=0)
                    elif label.data == 3:
                        tnum3 = tnum3 + 1
                        if tnum3 == 1:
                            all_data_target3 = all_data_tgt3
                            all_label_target3 = label
                        else:
                            all_data_target3 = torch.cat([all_data_target3, all_data_tgt3], dim=0)
                            all_label_target3 = torch.cat([all_label_target3, label], dim=0)
                    m = m + 1
                all_domain_t_loss0, all_domain_t_loss1, all_domain_t_loss3, all_domain_t_loss2 = 0, 0, 0, 0
                if tnum0 != 0:
                    t_label0 = Variable(torch.zeros(tnum0, 1).to(device))
                    all_domain_t_src0 = self.cls_cls10(all_data_target0, max_iter, 0)
                    all_domain_t_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_t_src0), t_label0)
                if tnum1 != 0:
                    t_label1 = Variable(torch.zeros(tnum1, 1).to(device))
                    all_domain_t_src1 = self.cls_cls11(all_data_target1, max_iter, 0)
                    all_domain_t_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_t_src1), t_label1)
                if tnum2 != 0:
                    t_label2 = Variable(torch.zeros(tnum2, 1).to(device))
                    all_domain_t_src2 = self.cls_cls12(all_data_target2, max_iter, 0)
                    all_domain_t_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_t_src2), t_label2)
                if tnum3 != 0:
                    t_label3 = Variable(torch.zeros(tnum3, 1).to(device))
                    all_domain_t_src3 = self.cls_cls13(all_data_target3, max_iter, 0)
                    all_domain_t_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_t_src3), t_label3)
                all_domain_t_loss = all_domain_t_loss0 + all_domain_t_loss1 + all_domain_t_loss3 + all_domain_t_loss2
                cls_loss_total = all_domain_src_loss + all_domain_t_loss
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                return cls_loss, domain_loss, l1_loss / 6,pl1_loss/6, cls_loss_total, data_src3, pred3, weight3_,pred_tgt_son3,data_src3
            if mark == 4:
                data_src4 = self.sonnet4(data_src)
                domain_pred_src = self.domain_cls4(data_src4, max_iter,1)
                # domain_src_loss = F.nll_loss(F.log_softmax(domain_pred_src, dim=1), s_domain_label)
                # domain_tgt_loss = F.nll_loss(F.log_softmax(data_tgt_son4, dim=1), t_domain_label)
                domain_src_loss = nn.BCELoss()(torch.sigmoid(domain_pred_src), s_domain_label)
                domain_tgt_loss = nn.BCELoss()(torch.sigmoid(data_tgt_son4), t_domain_label)
                domain_loss = domain_src_loss + domain_tgt_loss
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son4, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son4, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son4, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son3, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son4, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son5, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son4, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son6, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son4, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son7, dim=1)))
                pred_src = self.cls_fc_son4(data_src4)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pl1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(dis4, dim=1)
                                                - torch.nn.functional.softmax(dis1, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis4, dim=1)
                                                 - torch.nn.functional.softmax(dis2, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis4, dim=1)
                                                 - torch.nn.functional.softmax(dis3, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis4, dim=1)
                                                 - torch.nn.functional.softmax(dis5, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis4, dim=1)
                                                 - torch.nn.functional.softmax(dis6, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis4, dim=1)
                                                 - torch.nn.functional.softmax(dis7, dim=1)))
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # &&&&&&&&&&&&&&&&&&&&&&&1,2,3,4,5,6,7个源域的某个类看做一个大源域&&&&&&&&&&&&&&&&&&&&&&&&&
                j, m = 0, 0
                num0, num1, num2, num3 = 0, 0, 0, 0
                for label in label_src:
                    label = label.unsqueeze(0)
                    all_data_src4 = data_src4.data[j].unsqueeze(0)
                    if label.data == 0:
                        num0 = num0 + 1
                        if num0 == 1:
                            all_data_source0 = all_data_src4
                            all_label_source0 = label
                        else:
                            all_data_source0 = torch.cat([all_data_source0, all_data_src4], dim=0)
                            all_label_source0 = torch.cat([all_label_source0, label], dim=0)
                        # source_fea = self.cls_cls10(data_src1.data[j].to(device).to(device), max_iter)
                    elif label.data == 1:
                        num1 = num1 + 1
                        if num1 == 1:
                            all_data_source1 = all_data_src4
                            all_label_source1 = label
                        else:
                            all_data_source1 = torch.cat([all_data_source1, all_data_src4], dim=0)
                            all_label_source1 = torch.cat([all_label_source1, label], dim=0)
                        # source_fea = self.cls_cls11(data_src1.data[j].to(device), max_iter)
                    elif label.data == 2:
                        num2 = num2 + 1
                        # print(label.data)
                        # tensor([2], device='cuda:0')
                        # print(label)
                        # tensor([2], device='cuda:0')
                        # print(all_label_source0.size())
                        # torch.Size([1])
                        if num2 == 1:
                            all_data_source2 = all_data_src4
                            all_label_source2 = label
                            # print(all_label_source1)
                            # tensor([2], device='cuda:0')
                        else:
                            all_data_source2 = torch.cat([all_data_source2, all_data_src4], dim=0)
                            all_label_source2 = torch.cat([all_label_source2, label], dim=0)
                        # source_fea = self.cls_cls12(data_src1.data[j].to(device), max_iter)
                    elif label.data == 3:
                        num3 = num3 + 1
                        if num3 == 1:
                            all_data_source3 = all_data_src4
                            all_label_source3 = label
                        else:
                            all_data_source3 = torch.cat([all_data_source3, all_data_src4], dim=0)
                            all_label_source3 = torch.cat([all_label_source3, label], dim=0)
                        # source_fea = self.cls_cls13(data_src1.data[j].to(device), max_iter)
                    j = j + 1
                all_domain_src_loss0, all_domain_src_loss1, all_domain_src_loss3, all_domain_src_loss2 = 0, 0, 0, 0
                if num0 != 0:
                    s_label0 = Variable(torch.zeros(num0, 1).to(device))
                    all_domain_pred_src0 = self.cls_cls10(all_data_source0, max_iter, 0)
                    all_domain_src_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src0), s_label0)
                if num1 != 0:
                    s_label1 = Variable(torch.zeros(num1, 1).to(device))
                    all_domain_pred_src1 = self.cls_cls11(all_data_source1, max_iter, 0)
                    all_domain_src_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src1), s_label1)
                if num2 != 0:
                    s_label2 = Variable(torch.zeros(num2, 1).to(device))
                    all_domain_pred_src2 = self.cls_cls12(all_data_source2, max_iter, 0)
                    all_domain_src_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src2), s_label2)
                if num3 != 0:
                    s_label3 = Variable(torch.zeros(num3, 1).to(device))
                    all_domain_pred_src3 = self.cls_cls13(all_data_source3, max_iter, 0)
                    all_domain_src_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src3), s_label3)
                all_domain_src_loss = all_domain_src_loss0 + all_domain_src_loss1 + all_domain_src_loss3 + all_domain_src_loss2

                tnum0, tnum1, tnum2, tnum3 = 0, 0, 0, 0
                for label in pred4:
                    label = label.unsqueeze(0)
                    all_data_tgt4 = data_tgt4.data[m].unsqueeze(0)
                    if label.data == 0:
                        tnum0 = tnum0 + 1
                        if tnum0 == 1:
                            all_data_target0 = all_data_tgt4
                            all_label_target0 = label
                        else:
                            all_data_target0 = torch.cat([all_data_target0, all_data_tgt4], dim=0)
                            all_label_target0 = torch.cat([all_label_target0, label], dim=0)

                    elif label.data == 1:
                        tnum1 = tnum1 + 1
                        if tnum1 == 1:
                            all_data_target1 = all_data_tgt4
                            all_label_target1 = label
                        else:
                            all_data_target1 = torch.cat([all_data_target1, all_data_tgt4], dim=0)
                            all_label_target1 = torch.cat([all_label_target1, label], dim=0)
                    elif label.data == 2:
                        tnum2 = tnum2 + 1
                        if tnum2 == 1:
                            all_data_target2 = all_data_tgt4
                            all_label_target2 = label
                        else:
                            all_data_target2 = torch.cat([all_data_target2, all_data_tgt4], dim=0)
                            all_label_target2 = torch.cat([all_label_target2, label], dim=0)
                    elif label.data == 3:
                        tnum3 = tnum3 + 1
                        if tnum3 == 1:
                            all_data_target3 = all_data_tgt4
                            all_label_target3 = label
                        else:
                            all_data_target3 = torch.cat([all_data_target3, all_data_tgt4], dim=0)
                            all_label_target3 = torch.cat([all_label_target3, label], dim=0)
                    m = m + 1
                all_domain_t_loss0, all_domain_t_loss1, all_domain_t_loss3, all_domain_t_loss2 = 0, 0, 0, 0
                if tnum0 != 0:
                    t_label0 = Variable(torch.zeros(tnum0, 1).to(device))
                    all_domain_t_src0 = self.cls_cls10(all_data_target0, max_iter, 0)
                    all_domain_t_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_t_src0), t_label0)
                if tnum1 != 0:
                    t_label1 = Variable(torch.zeros(tnum1, 1).to(device))
                    all_domain_t_src1 = self.cls_cls11(all_data_target1, max_iter, 0)
                    all_domain_t_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_t_src1), t_label1)
                if tnum2 != 0:
                    t_label2 = Variable(torch.zeros(tnum2, 1).to(device))
                    all_domain_t_src2 = self.cls_cls12(all_data_target2, max_iter, 0)
                    all_domain_t_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_t_src2), t_label2)
                if tnum3 != 0:
                    t_label3 = Variable(torch.zeros(tnum3, 1).to(device))
                    all_domain_t_src3 = self.cls_cls13(all_data_target3, max_iter, 0)
                    all_domain_t_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_t_src3), t_label3)
                all_domain_t_loss = all_domain_t_loss0 + all_domain_t_loss1 + all_domain_t_loss3 + all_domain_t_loss2
                cls_loss_total = all_domain_src_loss + all_domain_t_loss
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                return cls_loss, domain_loss, l1_loss / 6, pl1_loss/6,cls_loss_total, data_src4, pred4,weight4_, pred_tgt_son4,data_src4
            if mark == 5:
                data_src5 = self.sonnet5(data_src)
                domain_pred_src = self.domain_cls5(data_src5, max_iter,1)
                # domain_src_loss = F.nll_loss(F.log_softmax(domain_pred_src, dim=1), s_domain_label)
                # domain_tgt_loss = F.nll_loss(F.log_softmax(data_tgt_son5, dim=1), t_domain_label)
                domain_src_loss = nn.BCELoss()(torch.sigmoid(domain_pred_src), s_domain_label)
                domain_tgt_loss = nn.BCELoss()(torch.sigmoid(data_tgt_son5), t_domain_label)
                domain_loss = domain_src_loss + domain_tgt_loss
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son5, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son5, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son5, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son3, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son5, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son4, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son5, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son6, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son5, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son7, dim=1)))
                pred_src = self.cls_fc_son5(data_src5)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pl1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(dis5, dim=1)
                                                - torch.nn.functional.softmax(dis1, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis5, dim=1)
                                                 - torch.nn.functional.softmax(dis2, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis5, dim=1)
                                                 - torch.nn.functional.softmax(dis3, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis5, dim=1)
                                                 - torch.nn.functional.softmax(dis4, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis5, dim=1)
                                                 - torch.nn.functional.softmax(dis6, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis5, dim=1)
                                                 - torch.nn.functional.softmax(dis7, dim=1)))
                # &&&&&&&&&&&&&&&&&&&&&&&1,2,3,4,5,6,7个源域的某个类看做一个大源域&&&&&&&&&&&&&&&&&&&&&&&&&
                j, m = 0, 0
                num0, num1, num2, num3 = 0, 0, 0, 0
                for label in label_src:
                    label = label.unsqueeze(0)
                    all_data_src5 = data_src5.data[j].unsqueeze(0)
                    if label.data == 0:
                        num0 = num0 + 1
                        if num0 == 1:
                            all_data_source0 = all_data_src5
                            all_label_source0 = label
                        else:
                            all_data_source0 = torch.cat([all_data_source0, all_data_src5], dim=0)
                            all_label_source0 = torch.cat([all_label_source0, label], dim=0)
                        # source_fea = self.cls_cls10(data_src1.data[j].to(device).to(device), max_iter)
                    elif label.data == 1:
                        num1 = num1 + 1
                        if num1 == 1:
                            all_data_source1 = all_data_src5
                            all_label_source1 = label
                        else:
                            all_data_source1 = torch.cat([all_data_source1, all_data_src5], dim=0)
                            all_label_source1 = torch.cat([all_label_source1, label], dim=0)
                        # source_fea = self.cls_cls11(data_src1.data[j].to(device), max_iter)
                    elif label.data == 2:
                        num2 = num2 + 1
                        # print(label.data)
                        # tensor([2], device='cuda:0')
                        # print(label)
                        # tensor([2], device='cuda:0')
                        # print(all_label_source0.size())
                        # torch.Size([1])
                        if num2 == 1:
                            all_data_source2 = all_data_src5
                            all_label_source2 = label
                            # print(all_label_source1)
                            # tensor([2], device='cuda:0')
                        else:
                            all_data_source2 = torch.cat([all_data_source2, all_data_src5], dim=0)
                            all_label_source2 = torch.cat([all_label_source2, label], dim=0)
                        # source_fea = self.cls_cls12(data_src1.data[j].to(device), max_iter)
                    elif label.data == 3:
                        num3 = num3 + 1
                        if num3 == 1:
                            all_data_source3 = all_data_src5
                            all_label_source3 = label
                        else:
                            all_data_source3 = torch.cat([all_data_source3, all_data_src5], dim=0)
                            all_label_source3 = torch.cat([all_label_source3, label], dim=0)
                        # source_fea = self.cls_cls13(data_src1.data[j].to(device), max_iter)
                    j = j + 1
                all_domain_src_loss0, all_domain_src_loss1, all_domain_src_loss3, all_domain_src_loss2 = 0, 0, 0, 0
                if num0 != 0:
                    s_label0 = Variable(torch.zeros(num0, 1).to(device))
                    all_domain_pred_src0 = self.cls_cls10(all_data_source0, max_iter, 0)
                    all_domain_src_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src0), s_label0)
                if num1 != 0:
                    s_label1 = Variable(torch.zeros(num1, 1).to(device))
                    all_domain_pred_src1 = self.cls_cls11(all_data_source1, max_iter, 0)
                    all_domain_src_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src1), s_label1)
                if num2 != 0:
                    s_label2 = Variable(torch.zeros(num2, 1).to(device))
                    all_domain_pred_src2 = self.cls_cls12(all_data_source2, max_iter, 0)
                    all_domain_src_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src2), s_label2)
                if num3 != 0:
                    s_label3 = Variable(torch.zeros(num3, 1).to(device))
                    all_domain_pred_src3 = self.cls_cls13(all_data_source3, max_iter, 0)
                    all_domain_src_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src3), s_label3)
                all_domain_src_loss = all_domain_src_loss0 + all_domain_src_loss1 + all_domain_src_loss3 + all_domain_src_loss2

                tnum0, tnum1, tnum2, tnum3 = 0, 0, 0, 0
                for label in pred5:
                    label = label.unsqueeze(0)
                    all_data_tgt5 = data_tgt5.data[m].unsqueeze(0)
                    if label.data == 0:
                        tnum0 = tnum0 + 1
                        if tnum0 == 1:
                            all_data_target0 = all_data_tgt5
                            all_label_target0 = label
                        else:
                            all_data_target0 = torch.cat([all_data_target0, all_data_tgt5], dim=0)
                            all_label_target0 = torch.cat([all_label_target0, label], dim=0)

                    elif label.data == 1:
                        tnum1 = tnum1 + 1
                        if tnum1 == 1:
                            all_data_target1 = all_data_tgt5
                            all_label_target1 = label
                        else:
                            all_data_target1 = torch.cat([all_data_target1, all_data_tgt5], dim=0)
                            all_label_target1 = torch.cat([all_label_target1, label], dim=0)
                    elif label.data == 2:
                        tnum2 = tnum2 + 1
                        if tnum2 == 1:
                            all_data_target2 = all_data_tgt5
                            all_label_target2 = label
                        else:
                            all_data_target2 = torch.cat([all_data_target2, all_data_tgt5], dim=0)
                            all_label_target2 = torch.cat([all_label_target2, label], dim=0)
                    elif label.data == 3:
                        tnum3 = tnum3 + 1
                        if tnum3 == 1:
                            all_data_target3 = all_data_tgt5
                            all_label_target3 = label
                        else:
                            all_data_target3 = torch.cat([all_data_target3, all_data_tgt5], dim=0)
                            all_label_target3 = torch.cat([all_label_target3, label], dim=0)
                    m = m + 1
                all_domain_t_loss0, all_domain_t_loss1, all_domain_t_loss3, all_domain_t_loss2 = 0, 0, 0, 0
                if tnum0 != 0:
                    t_label0 = Variable(torch.zeros(tnum0, 1).to(device))
                    all_domain_t_src0 = self.cls_cls10(all_data_target0, max_iter, 0)
                    all_domain_t_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_t_src0), t_label0)
                if tnum1 != 0:
                    t_label1 = Variable(torch.zeros(tnum1, 1).to(device))
                    all_domain_t_src1 = self.cls_cls11(all_data_target1, max_iter, 0)
                    all_domain_t_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_t_src1), t_label1)
                if tnum2 != 0:
                    t_label2 = Variable(torch.zeros(tnum2, 1).to(device))
                    all_domain_t_src2 = self.cls_cls12(all_data_target2, max_iter, 0)
                    all_domain_t_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_t_src2), t_label2)
                if tnum3 != 0:
                    t_label3 = Variable(torch.zeros(tnum3, 1).to(device))
                    all_domain_t_src3 = self.cls_cls13(all_data_target3, max_iter, 0)
                    all_domain_t_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_t_src3), t_label3)
                all_domain_t_loss = all_domain_t_loss0 + all_domain_t_loss1 + all_domain_t_loss3 + all_domain_t_loss2
                cls_loss_total = all_domain_src_loss + all_domain_t_loss
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                return cls_loss, domain_loss, l1_loss / 6, pl1_loss/6,cls_loss_total, data_src5, pred5,weight5_, pred_tgt_son5,data_src5
            if mark == 6:
                data_src6 = self.sonnet6(data_src)
                domain_pred_src = self.domain_cls6(data_src6, max_iter,1)
                # domain_src_loss = F.nll_loss(F.log_softmax(domain_pred_src, dim=1), s_domain_label)
                # domain_tgt_loss = F.nll_loss(F.log_softmax(data_tgt_son6, dim=1), t_domain_label)
                domain_src_loss = nn.BCELoss()(torch.sigmoid(domain_pred_src), s_domain_label)
                domain_tgt_loss = nn.BCELoss()(torch.sigmoid(data_tgt_son6), t_domain_label)
                domain_loss = domain_src_loss + domain_tgt_loss

                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son6, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son6, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son6, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son3, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son6, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son4, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son6, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son5, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son6, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son7, dim=1)))
                pred_src = self.cls_fc_son6(data_src6)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pl1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(dis6, dim=1)
                                                - torch.nn.functional.softmax(dis1, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis6, dim=1)
                                                 - torch.nn.functional.softmax(dis2, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis6, dim=1)
                                                 - torch.nn.functional.softmax(dis3, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis6, dim=1)
                                                 - torch.nn.functional.softmax(dis4, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis6, dim=1)
                                                 - torch.nn.functional.softmax(dis5, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis6, dim=1)
                                                 - torch.nn.functional.softmax(dis7, dim=1)))
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # &&&&&&&&&&&&&&&&&&&&&&&1,2,3,4,5,6,7个源域的某个类看做一个大源域&&&&&&&&&&&&&&&&&&&&&&&&&
                j, m = 0, 0
                num0, num1, num2, num3 = 0, 0, 0, 0
                for label in label_src:
                    label = label.unsqueeze(0)
                    all_data_src6 = data_src6.data[j].unsqueeze(0)
                    if label.data == 0:
                        num0 = num0 + 1
                        if num0 == 1:
                            all_data_source0 = all_data_src6
                            all_label_source0 = label
                        else:
                            all_data_source0 = torch.cat([all_data_source0, all_data_src6], dim=0)
                            all_label_source0 = torch.cat([all_label_source0, label], dim=0)
                        # source_fea = self.cls_cls10(data_src1.data[j].to(device).to(device), max_iter)
                    elif label.data == 1:
                        num1 = num1 + 1
                        if num1 == 1:
                            all_data_source1 = all_data_src6
                            all_label_source1 = label
                        else:
                            all_data_source1 = torch.cat([all_data_source1, all_data_src6], dim=0)
                            all_label_source1 = torch.cat([all_label_source1, label], dim=0)
                        # source_fea = self.cls_cls11(data_src1.data[j].to(device), max_iter)
                    elif label.data == 2:
                        num2 = num2 + 1
                        # print(label.data)
                        # tensor([2], device='cuda:0')
                        # print(label)
                        # tensor([2], device='cuda:0')
                        # print(all_label_source0.size())
                        # torch.Size([1])
                        if num2 == 1:
                            all_data_source2 = all_data_src6
                            all_label_source2 = label
                            # print(all_label_source1)
                            # tensor([2], device='cuda:0')
                        else:
                            all_data_source2 = torch.cat([all_data_source2, all_data_src6], dim=0)
                            all_label_source2 = torch.cat([all_label_source2, label], dim=0)
                        # source_fea = self.cls_cls12(data_src1.data[j].to(device), max_iter)
                    elif label.data == 3:
                        num3 = num3 + 1
                        if num3 == 1:
                            all_data_source3 = all_data_src6
                            all_label_source3 = label
                        else:
                            all_data_source3 = torch.cat([all_data_source3, all_data_src6], dim=0)
                            all_label_source3 = torch.cat([all_label_source3, label], dim=0)
                        # source_fea = self.cls_cls13(data_src1.data[j].to(device), max_iter)
                    j = j + 1
                all_domain_src_loss0, all_domain_src_loss1, all_domain_src_loss3, all_domain_src_loss2 = 0, 0, 0, 0
                if num0 != 0:
                    s_label0 = Variable(torch.zeros(num0, 1).to(device))
                    all_domain_pred_src0 = self.cls_cls10(all_data_source0, max_iter, 0)
                    all_domain_src_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src0), s_label0)
                if num1 != 0:
                    s_label1 = Variable(torch.zeros(num1, 1).to(device))
                    all_domain_pred_src1 = self.cls_cls11(all_data_source1, max_iter, 0)
                    all_domain_src_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src1), s_label1)
                if num2 != 0:
                    s_label2 = Variable(torch.zeros(num2, 1).to(device))
                    all_domain_pred_src2 = self.cls_cls12(all_data_source2, max_iter, 0)
                    all_domain_src_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src2), s_label2)
                if num3 != 0:
                    s_label3 = Variable(torch.zeros(num3, 1).to(device))
                    all_domain_pred_src3 = self.cls_cls13(all_data_source3, max_iter, 0)
                    all_domain_src_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src3), s_label3)
                all_domain_src_loss = all_domain_src_loss0 + all_domain_src_loss1 + all_domain_src_loss3 + all_domain_src_loss2

                tnum0, tnum1, tnum2, tnum3 = 0, 0, 0, 0
                for label in pred6:
                    label = label.unsqueeze(0)
                    all_data_tgt6 = data_tgt6.data[m].unsqueeze(0)
                    if label.data == 0:
                        tnum0 = tnum0 + 1
                        if tnum0 == 1:
                            all_data_target0 = all_data_tgt6
                            all_label_target0 = label
                        else:
                            all_data_target0 = torch.cat([all_data_target0, all_data_tgt6], dim=0)
                            all_label_target0 = torch.cat([all_label_target0, label], dim=0)

                    elif label.data == 1:
                        tnum1 = tnum1 + 1
                        if tnum1 == 1:
                            all_data_target1 = all_data_tgt6
                            all_label_target1 = label
                        else:
                            all_data_target1 = torch.cat([all_data_target1, all_data_tgt6], dim=0)
                            all_label_target1 = torch.cat([all_label_target1, label], dim=0)
                    elif label.data == 2:
                        tnum2 = tnum2 + 1
                        if tnum2 == 1:
                            all_data_target2 = all_data_tgt6
                            all_label_target2 = label
                        else:
                            all_data_target2 = torch.cat([all_data_target2, all_data_tgt6], dim=0)
                            all_label_target2 = torch.cat([all_label_target2, label], dim=0)
                    elif label.data == 3:
                        tnum3 = tnum3 + 1
                        if tnum3 == 1:
                            all_data_target3 = all_data_tgt6
                            all_label_target3 = label
                        else:
                            all_data_target3 = torch.cat([all_data_target3, all_data_tgt6], dim=0)
                            all_label_target3 = torch.cat([all_label_target3, label], dim=0)
                    m = m + 1
                all_domain_t_loss0, all_domain_t_loss1, all_domain_t_loss3, all_domain_t_loss2 = 0, 0, 0, 0
                if tnum0 != 0:
                    t_label0 = Variable(torch.zeros(tnum0, 1).to(device))
                    all_domain_t_src0 = self.cls_cls10(all_data_target0, max_iter, 0)
                    all_domain_t_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_t_src0), t_label0)
                if tnum1 != 0:
                    t_label1 = Variable(torch.zeros(tnum1, 1).to(device))
                    all_domain_t_src1 = self.cls_cls11(all_data_target1, max_iter, 0)
                    all_domain_t_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_t_src1), t_label1)
                if tnum2 != 0:
                    t_label2 = Variable(torch.zeros(tnum2, 1).to(device))
                    all_domain_t_src2 = self.cls_cls12(all_data_target2, max_iter, 0)
                    all_domain_t_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_t_src2), t_label2)
                if tnum3 != 0:
                    t_label3 = Variable(torch.zeros(tnum3, 1).to(device))
                    all_domain_t_src3 = self.cls_cls13(all_data_target3, max_iter, 0)
                    all_domain_t_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_t_src3), t_label3)
                all_domain_t_loss = all_domain_t_loss0 + all_domain_t_loss1 + all_domain_t_loss3 + all_domain_t_loss2
                cls_loss_total = all_domain_src_loss + all_domain_t_loss
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                return cls_loss, domain_loss, l1_loss / 6, pl1_loss/6,cls_loss_total, data_src6, pred6,weight6_, pred_tgt_son6,data_src6
            if mark == 7:
                data_src7 = self.sonnet7(data_src)
                domain_pred_src = self.domain_cls7(data_src7, max_iter,1)
                # domain_src_loss = F.nll_loss(F.log_softmax(domain_pred_src, dim=1), s_domain_label)
                # domain_tgt_loss = F.nll_loss(F.log_softmax(data_tgt_son7, dim=1), t_domain_label)
                domain_src_loss = nn.BCELoss()(torch.sigmoid(domain_pred_src), s_domain_label)
                domain_tgt_loss = nn.BCELoss()(torch.sigmoid(data_tgt_son7), t_domain_label)
                domain_loss = domain_src_loss + domain_tgt_loss
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son7, dim=1)
                                               - torch.nn.functional.softmax(pred_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son7, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son7, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son3, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son7, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son4, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son7, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son5, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(pred_tgt_son7, dim=1)
                                                - torch.nn.functional.softmax(pred_tgt_son6, dim=1)))
                pred_src = self.cls_fc_son7(data_src7)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # ￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
                pl1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(dis7, dim=1)
                                                - torch.nn.functional.softmax(dis1, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis7, dim=1)
                                                 - torch.nn.functional.softmax(dis2, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis7, dim=1)
                                                 - torch.nn.functional.softmax(dis3, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis7, dim=1)
                                                 - torch.nn.functional.softmax(dis4, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis7, dim=1)
                                                 - torch.nn.functional.softmax(dis5, dim=1)))
                pl1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(dis7, dim=1)
                                                 - torch.nn.functional.softmax(dis6, dim=1)))
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # &&&&&&&&&&&&&&&&&&&&&&&1,2,3,4,5,6,7个源域的某个类看做一个大源域&&&&&&&&&&&&&&&&&&&&&&&&&
                j, m = 0, 0
                num0, num1, num2, num3 = 0, 0, 0, 0
                for label in label_src:
                    label = label.unsqueeze(0)
                    all_data_src7 = data_src7.data[j].unsqueeze(0)
                    if label.data == 0:
                        num0 = num0 + 1
                        if num0 == 1:
                            all_data_source0 = all_data_src7
                            all_label_source0 = label
                        else:
                            all_data_source0 = torch.cat([all_data_source0, all_data_src7], dim=0)
                            all_label_source0 = torch.cat([all_label_source0, label], dim=0)
                        # source_fea = self.cls_cls10(data_src1.data[j].to(device).to(device), max_iter)
                    elif label.data == 1:
                        num1 = num1 + 1
                        if num1 == 1:
                            all_data_source1 = all_data_src7
                            all_label_source1 = label
                        else:
                            all_data_source1 = torch.cat([all_data_source1, all_data_src7], dim=0)
                            all_label_source1 = torch.cat([all_label_source1, label], dim=0)
                        # source_fea = self.cls_cls11(data_src1.data[j].to(device), max_iter)
                    elif label.data == 2:
                        num2 = num2 + 1
                        # print(label.data)
                        # tensor([2], device='cuda:0')
                        # print(label)
                        # tensor([2], device='cuda:0')
                        # print(all_label_source0.size())
                        # torch.Size([1])
                        if num2 == 1:
                            all_data_source2 = all_data_src7
                            all_label_source2 = label
                            # print(all_label_source1)
                            # tensor([2], device='cuda:0')
                        else:
                            all_data_source2 = torch.cat([all_data_source2, all_data_src7], dim=0)
                            all_label_source2 = torch.cat([all_label_source2, label], dim=0)
                        # source_fea = self.cls_cls12(data_src1.data[j].to(device), max_iter)
                    elif label.data == 3:
                        num3 = num3 + 1
                        if num3 == 1:
                            all_data_source3 = all_data_src7
                            all_label_source3 = label
                        else:
                            all_data_source3 = torch.cat([all_data_source3, all_data_src7], dim=0)
                            all_label_source3 = torch.cat([all_label_source3, label], dim=0)
                        # source_fea = self.cls_cls13(data_src1.data[j].to(device), max_iter)
                    j = j + 1
                all_domain_src_loss0, all_domain_src_loss1, all_domain_src_loss3, all_domain_src_loss2 = 0, 0, 0, 0
                if num0 != 0:
                    s_label0 = Variable(torch.zeros(num0, 1).to(device))
                    all_domain_pred_src0 = self.cls_cls10(all_data_source0, max_iter, 0)
                    all_domain_src_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src0), s_label0)
                if num1 != 0:
                    s_label1 = Variable(torch.zeros(num1, 1).to(device))
                    all_domain_pred_src1 = self.cls_cls11(all_data_source1, max_iter, 0)
                    all_domain_src_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src1), s_label1)
                if num2 != 0:
                    s_label2 = Variable(torch.zeros(num2, 1).to(device))
                    all_domain_pred_src2 = self.cls_cls12(all_data_source2, max_iter, 0)
                    all_domain_src_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src2), s_label2)
                if num3 != 0:
                    s_label3 = Variable(torch.zeros(num3, 1).to(device))
                    all_domain_pred_src3 = self.cls_cls13(all_data_source3, max_iter, 0)
                    all_domain_src_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_pred_src3), s_label3)
                all_domain_src_loss = all_domain_src_loss0 + all_domain_src_loss1 + all_domain_src_loss3 + all_domain_src_loss2

                tnum0, tnum1, tnum2, tnum3 = 0, 0, 0, 0
                for label in pred7:
                    label = label.unsqueeze(0)
                    all_data_tgt7 = data_tgt7.data[m].unsqueeze(0)
                    if label.data == 0:
                        tnum0 = tnum0 + 1
                        if tnum0 == 1:
                            all_data_target0 = all_data_tgt7
                            all_label_target0 = label
                        else:
                            all_data_target0 = torch.cat([all_data_target0, all_data_tgt7], dim=0)
                            all_label_target0 = torch.cat([all_label_target0, label], dim=0)

                    elif label.data == 1:
                        tnum1 = tnum1 + 1
                        if tnum1 == 1:
                            all_data_target1 = all_data_tgt7
                            all_label_target1 = label
                        else:
                            all_data_target1 = torch.cat([all_data_target1, all_data_tgt7], dim=0)
                            all_label_target1 = torch.cat([all_label_target1, label], dim=0)
                    elif label.data == 2:
                        tnum2 = tnum2 + 1
                        if tnum2 == 1:
                            all_data_target2 = all_data_tgt7
                            all_label_target2 = label
                        else:
                            all_data_target2 = torch.cat([all_data_target2, all_data_tgt7], dim=0)
                            all_label_target2 = torch.cat([all_label_target2, label], dim=0)
                    elif label.data == 3:
                        tnum3 = tnum3 + 1
                        if tnum3 == 1:
                            all_data_target3 = all_data_tgt7
                            all_label_target3 = label
                        else:
                            all_data_target3 = torch.cat([all_data_target3, all_data_tgt7], dim=0)
                            all_label_target3 = torch.cat([all_label_target3, label], dim=0)
                    m = m + 1
                all_domain_t_loss0, all_domain_t_loss1, all_domain_t_loss3, all_domain_t_loss2 = 0, 0, 0, 0
                if tnum0 != 0:
                    t_label0 = Variable(torch.zeros(tnum0, 1).to(device))
                    all_domain_t_src0 = self.cls_cls10(all_data_target0, max_iter, 0)
                    all_domain_t_loss0 = nn.BCELoss()(torch.sigmoid(all_domain_t_src0), t_label0)
                if tnum1 != 0:
                    t_label1 = Variable(torch.zeros(tnum1, 1).to(device))
                    all_domain_t_src1 = self.cls_cls11(all_data_target1, max_iter, 0)
                    all_domain_t_loss1 = nn.BCELoss()(torch.sigmoid(all_domain_t_src1), t_label1)
                if tnum2 != 0:
                    t_label2 = Variable(torch.zeros(tnum2, 1).to(device))
                    all_domain_t_src2 = self.cls_cls12(all_data_target2, max_iter, 0)
                    all_domain_t_loss2 = nn.BCELoss()(torch.sigmoid(all_domain_t_src2), t_label2)
                if tnum3 != 0:
                    t_label3 = Variable(torch.zeros(tnum3, 1).to(device))
                    all_domain_t_src3 = self.cls_cls13(all_data_target3, max_iter, 0)
                    all_domain_t_loss3 = nn.BCELoss()(torch.sigmoid(all_domain_t_src3), t_label3)
                all_domain_t_loss = all_domain_t_loss0 + all_domain_t_loss1 + all_domain_t_loss3 + all_domain_t_loss2
                cls_loss_total = all_domain_src_loss + all_domain_t_loss
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                return cls_loss, domain_loss, l1_loss / 6,pl1_loss / 6, cls_loss_total, data_src7, pred7,weight7_, pred_tgt_son7,data_src7

        else:
            data = self.sharedNet(data_src)
            features_target1 = self.sonnet1(data)
            pred1 = self.cls_fc_son1(features_target1)
            features_target2 = self.sonnet2(data)
            pred2 = self.cls_fc_son2(features_target2)
            features_target3 = self.sonnet3(data)
            pred3 = self.cls_fc_son3(features_target3)
            features_target4 = self.sonnet4(data)
            pred4 = self.cls_fc_son4(features_target4)
            features_target5 = self.sonnet5(data)
            pred5 = self.cls_fc_son5(features_target5)
            features_target6 = self.sonnet6(data)
            pred6 = self.cls_fc_son6(features_target6)
            features_target7 = self.sonnet7(data)
            pred7 = self.cls_fc_son7(features_target7)
            return pred1, pred2, pred3, pred4, pred5, pred6, pred7, features_target1, features_target2, features_target3, features_target4, features_target5, features_target6, features_target7


class classification(nn.Module):
    def __init__(self, sent_hidden_size):
        super(classification, self).__init__()
        self.fc1=nn.Linear(sent_hidden_size, sent_hidden_size)
        self.bn1=nn.BatchNorm1d(sent_hidden_size)
        self.relu1=nn.ReLU(True)
        self.droput= nn.Dropout()
        self.fc2=nn.Linear(sent_hidden_size, 4, bias=False)
    def forward(self, x):
        # print(x.size())---torch.Size([8, 100])
        output=self.fc1(x)
        output=self.bn1(output)
        output=self.relu1(output)
        # output=self.droput(output)
        output=self.fc2(output)
        # print(output.size())----torch.Size([8, 4])
        # print(sss)
        return output



