import torch
import torch.nn as nn
import torch.nn.functional as F
#SentAttNet(sent_hidden_size, word_hidden_size)
class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=100, word_hidden_size=100):
        super(SentAttNet, self).__init__()
        #word_hidden_size=100表示每一个单词的维度为100
        self.LSTM = nn.LSTM(word_hidden_size, sent_hidden_size)
        self.fc1 = nn.Linear( sent_hidden_size,sent_hidden_size)
        self.fc2 = nn.Linear(sent_hidden_size,1, bias=False)

    def forward(self, input):
        #input(seq_len, batch_size, input_size)---（97,5,100）
        #输入由两部分组成：input、(初始的隐状态h_0，初始的单元状态c_0)
        # h_0(num_directions * num_layers, batch_size, hidden_size)---（1,5,100）
        # c_0(num_directions * num_layers, batch_size, hidden_size)---（1,5,100）
        #输出也由两部分组成：otput、(隐状态h_n，单元状态c_n)
        #output(seq_len, batch_size, num_directions * hidden_size)---（97,5,100）
        # print(input.size())---torch.Size([97, 5, 100])
        #（97,5,100）----需要处理的数据：5篇文章，每篇文章有97个句子，每个句子用100个单词表示
        self.LSTM.flatten_parameters()
        f_output, _ = self.LSTM(input)
        # print(f_output.size())---torch.Size([97, 5, 100])
        #LSTM得到当前时刻的ht和Ot，使用当前时刻的隐藏状态ht
        weight = torch.tanh(self.fc1(f_output))
        #将 ht输入到一个 dense 网络中得到的结果ut作为ht的隐含表示。
        weight = self.fc2(weight)
        weight = F.softmax(weight,0)
        #用 ut和一个随机初始化的上下文向量uw的相似度来表示，然后经过 softmax 操作获得了一个归一化的 attention 权重矩阵a，代表某个句子中第 t个词的权重。
        weight = weight * f_output
        # print(weight.size())----torch.Size([97, 8, 100])
        # 权重矩阵加权求和
        output = weight.sum(0).unsqueeze(0)
        # print(output.size())---torch.Size([1, 5, 100])
        return output


if __name__ == "__main__":
    abc = SentAttNet()
