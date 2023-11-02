import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#WordAttNet(embed_table,word_hidden_size)
class WordAttNet(nn.Module):
    def __init__(self, dict,hidden_size=100):
        super(WordAttNet, self).__init__()
        # print(dict.dtype)----->float64
        # print(dict.astype(np.float).dtype)---->float64
        # dict.astype(np.float)将dict转换数据类型
        #torch.from_numpy(ndarray)的作用就是将生成的数组（数组是一般理解，标准应该称为array）转换为张量Tensor，此函数在数字图像处理中应用广泛，尤其是在训练图像数据集时候会经常看到。
        dict = torch.from_numpy(dict.astype(np.float))
        # print(dict)
        # print(sss)
        # print(dict.dtype)---->torch.float64
        # pytorch中的常见的Tensor数据类型，例如：float32，float64，int32，int64。
        # 构造他们分别使用如下函数：torch.FloatTensor()；torch.DoubleTensor(), torch.IntTensor(), torch.LongTensor()。
        # torch.float64对应torch.DoubleTensor
        #使用nn.Embedding.from_pretrained()加载预训练好的模型，如word2vec,glove等
        self.lookup = nn.Embedding(num_embeddings=5000, embedding_dim=50).from_pretrained(dict)
        #字典长度=4000
        self.conv1 = nn.Conv1d(in_channels=50,out_channels=100,kernel_size=5)
        # self.bn1=nn.BatchNorm1d(100)
        #in_channels(int) – 输入信号的通道。在文本分类中，即为词向量的维度，
        #out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
        #kernel_size(int or tuple) - 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear( 100,100)
        self.fc2 = nn.Linear( 100,1,bias =False)


    def forward(self, input):
        # print("进入lookup即embedding层：")
        output = self.lookup(input)
        # print("Embedding output.size():{}".format(output.size()))--torch.Size([5, sequence_length=50, output_dim=50])
        # 输入shape
        # 形如（samples，sequence_length）的2D张量
        # 输出shape
        # 形如(samples, sequence_length, output_dim)
        # 的3D张量
        output = self.dropout(output)
        #下面这一句是原来的
        #output = output.permute(1,2,0)
        #下面这一句是我修改的
        output = output.permute(0,2,1)
        # print(output.size())---（5,50,50）
        # 输入：（批大小， 数据的通道数， 数据长度）---（5,50,50）
        #输出：（批大小， 产生的通道数， 卷积后长度）---（5,100,46）
        f_output = self.conv1(output.float()) # shape : batch * hidden_size * seq_len
        # print("卷积后：{}".format(f_output.size()))---卷积后：torch.Size([5, 100, 46])
        f_output = f_output.permute(2,0,1)   # shape : seq_len * batch * hidden_size
        # print("permute(2,0,1) 后：{}".format(f_output.size()))---torch.Size([46, 5, 100])
        #卷积并经过一层线性层得到隐向量后开始注意力，tanh得到h的隐含表示（下面一行）U
        weight = torch.tanh(self.fc1(f_output))
        #衡量单词重要性后经过softmax得到权重α（下面两行）
        # print(weight.size())----torch.Size([46, 8, 100])
        weight = self.fc2(weight)
        # print(weight.size())----torch.Size([46, 8, 1])
        weight = F.softmax(weight,0)
        # print(weight.size())---torch.Size([46, 8, 1])
        #得到权重矩阵后加权求和得到句子向量（下面两行）
        weight = weight * f_output
        # print(weight.size())---torch.Size([46, 5, 100])
        output = weight.sum(0).unsqueeze(0)  # 1 * batch * hidden_size
        # print("得到句子向量：{}".format(output.size()))---torch.Size([1, 5, 100])
        return output


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
