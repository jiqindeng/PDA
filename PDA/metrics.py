#coding = utf-8
import numpy as np

def pearson(pred_score, true_score):

    pred_avg = np.average(pred_score)
    true_avg = np.average(true_score)

    num, n1, n2 = 0.0, 0.0, 0.0
    for pred_t, true_t in zip(pred_score, true_score):
        num += (pred_t - pred_avg) * (true_t - true_avg)
        n1 += (pred_t - pred_avg) * (pred_t - pred_avg)
        n2 += (true_t - true_avg) * (true_t - true_avg)

    return num / np.power(n1 * n2, 0.5)


def spearman(pred_score, true_score):

    pred_score = np.asarray(pred_score)
    true_score = np.asarray(true_score)

    pred_sort = np.sort(pred_score)

    true_sort = np.sort(true_score)

    pred_index, true_index = [], []
    for pred_t, true_t in zip(pred_score, true_score):
        index_list = np.where(pred_sort == pred_t)

        index = (index_list[0] + index_list[-1]) / 2

        pred_index.append(index[0])

        index_list = np.where(true_sort == true_t)
        index = (index_list[0] + index_list[-1]) / 2

        true_index.append(index[0])

    nb = len(pred_score)
    err = 0.0
    for pred_i, true_i in zip(pred_index, true_index):
        err += np.power(pred_i - true_i, 2)

    return 1.0 - 6.0 * err / (np.power(nb, 3) - nb)

#conf_mat = confusion_matrix(rater_a, rater_b,min_rating, max_rating)
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

#histogram(rater_a, min_rating, max_rating)
#rater_a=[9.  8.  8.  9.  9.  8.  8.  9.  8.  8.]
#min_rating=6, max_rating=11
def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    #num_ratings=6
    hist_ratings = [0 for x in range(num_ratings)]
    #hist_ratings=[0, 0, 0, 0, 0, 0]
    for r in ratings:
        # print(r)--->第一次循环取得是rater_a的第一个值：9
        hist_ratings[r - min_rating] += 1
    # print(hist_ratings)--->[0, 0, 6, 4, 0, 0]
    # hist_ratings[2]处为6表示预测结果为8的数据有6条(一共10条数据)
    return hist_ratings
#quadratic_weighted_kappa(predictions, te_label)
# print(predictions)--->dev的真实标签没有转为0-1之间，但是预测标签是在0-1之间的，所以预测的标签需要转为原始分数范围
            # [9.  8.  8.  9.  9.  8.  8.  9.  8.  8.]
# print(te_label)---->类似于下面这个样子
            # [9.  6.  8.  8. 11.  9.  8.  8. 11.  8.]
def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    # print(min_rating)----0
    # print(max_rating)-----3
    # 如果采用上面的预测值和真实值例子，min_rating=6，max_rating=11.此时矩阵为11-6+1=6,6行6列矩阵---》因为取值范围是6,7,8,9,10,11这6个种类
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    # print(conf_mat)
    #   0    1    2     3     真实为0却预测为2的有116个，真实1预测为2的有530个.....
#   0 [[0,   0,   0,   0],
#   1  [0,   0,   0,   0],
#   2  [116, 530, 550, 64],
#   3  [0,   0,   0,   0]]
#     print(ssss)
    #conf_mat类似下面这种：
   # [[0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0],
    # [1, 0, 3, 1, 0, 1],
    # [0, 0, 2, 1, 0, 1],
    # [0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0]]
    #行从下标0到5分别表示种类6,7,8,9,10,11
    #列从下标0到5分别表示种类6,7,8,9,10,11
    #例如预测list和真实list的第一个数都是9（正确分类：在conf_mat[3][3]处即预测为9真实为9处数量+1）
    #例如预测list和真实list的第二个数分别是8,6（错误分类：在conf_mat[2][0]处即预测为8真实为6处数量+1]）
    #依次类推就构造了混淆矩阵conf_mat
    #例如我们看到conf_mat[2][2]处为3说明有3个分类为8的数据正确分类为8了
    # 例如我们看到conf_mat[2][5]处为1说明有1个真实分类为11的数据预测的类别为8
    num_ratings = len(conf_mat)
    # print(num_ratings)------4
    #在举的例子里num_ratings=6
    num_scored_items = float(len(rater_a))
    # print(num_scored_items)-----1260
    #在举的例子里num_scored_items=10.0
    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    # hist_rater_a:
    # [0, 0, 1260, 0]
    #假设rater_a=[9 8 8 9 9 8 8 9 8 8]
    # hist_rater_a=[0, 0, 6, 4, 0, 0]
    #例如：hist_ratings[2]处为6表示预测结果为8的数据有6条(一共10条数据)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)
    # print(hist_rater_b)----[116, 530, 550, 64]
    #rater_b=[ 9  6  8  8 11  9  8  8 11  8]
    #hist_rater_b=[1, 0, 5, 2, 0, 2]
    # 例如：hist_ratings[0]处为1表示真实标签为6的数据有1条(一共10条数据)

    numerator = 0.0
    denominator = 0.0
    if num_ratings!=1:
        for i in range(num_ratings):
            for j in range(num_ratings):
                expected_count = (hist_rater_a[i] * hist_rater_b[j]
                                  / num_scored_items)
                d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
                numerator += d * conf_mat[i][j] / num_scored_items
                denominator += d * expected_count / num_scored_items
    else:
        numerator=0.0
        denominator=1.0
    return 1.0 - numerator / denominator
