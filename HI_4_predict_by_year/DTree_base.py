import csv
import json
import torch
import numpy as np
import xlrd
import random
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from xlwings import xrange
from sklearn.tree import DecisionTreeClassifier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Perceptron(object):
    def __init__(self):
        self.learning_step = 0.0001
        self.max_iteration = 5000

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in xrange(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)  #权重+偏置
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = np.random.randint(0, len(labels) - 1) #随机找个样本
            x = list(features[index])
            x.append(1)  #偏置
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            #错误就更新参数
            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])


    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels
#定义随机种子，用于数据集的抽样
random_seed = 24


#该文件根据dataset_construct产生的dna_HI_titler_dataset.xlsx生成k_mer数据集
class ReadExcel:
    def __init__(self, file_path):
        self.file_path = file_path

    def readExcel(self, sheet_index):
        # open the excel file
        table = xlrd.open_workbook(self.file_path)
        # choose the sheet to open
        data = table.sheet_by_name(table.sheet_names()[sheet_index])
        rows = data.nrows
        cols = data.ncols
        headers = data.row(0)

        table_matrix = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                cell_value = data.cell(i, j).value
                if cell_value == "":
                    cell_value = "NAN"
                row_data.append(cell_value)
            table_matrix.append(row_data)
        return table_matrix, headers

#获取到每一个基因与模板基因的开始位置的距离。
# 模板基因序列为   A T C T T T T A A
#       一个基因       C T T T A A A
#diff_position 即为2。
#diff_pos存储的所有基因序列与最长基因序列的开始位置的距离
def getDiffDic():
    with open("../diff_pos.json", 'r', encoding='UTF-8') as f:
        reduce_dict = json.load(f)
        return reduce_dict


# we fix the seeds to get consistent results before every training
# loop in what follows
def fix_seed(seed=13):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

#获取句子的embedding表示
# def getSeqEmbedding(seq):
#     sentence_embedding = torch.ones_like(word_embeddings_dic["[PAD]"])
#     for kmer in seq.split(' '):
#         if kmer in word_embeddings_dic.keys():
#             sentence_embedding = word_embeddings_dic[kmer] + sentence_embedding
#         else:
#             sentence_embedding = sentence_embedding + torch.ones_like(word_embeddings_dic['[PAD]'])
#
#     sentence_embedding = 1/len(seq.split(' '))*sentence_embedding
#     return sentence_embedding

#word2idx

def get_kmer2idx(vocab):
    word2idx = {w: idx for (idx, w) in enumerate(vocab)}
    return word2idx

#返回kmer列表，形如[[seq1, seq2, label],...]的列表
def getKmer(data_path):
    kmer_pairs = []
    with open(data_path, "r", encoding="utf-8") as r:
        kmer_data = r.readlines()
        i = 0
        for kmer in kmer_data:
            if i == 0 :
                i = i + 1
                continue
            tripules = kmer.split("\t")
            seq1 = tripules[0]
            seq2 = tripules[1]
            label = tripules[2]
            virus_1 = tripules[3]
            virus_2 = tripules[4]
            kmer_pairs.append([seq1, seq2, label, virus_1, virus_2])
    return kmer_pairs

#返回kmer的词向量词典
def get_word_embedding_dict(vocab_path, weight_parameter):
    idx2emb_dict = {}
    with open(vocab_path, "r", encoding="utf-8") as vocab:
        vocabulary = vocab.readlines()
        words = []
        for word in vocabulary:
            words.append(word.replace("\n", ""))

    word_embedding_dic = {}
    for word_index in range(len(words)):
        word_embedding_dic[words[word_index]] = weight_parameter[word_index]
        idx2emb_dict[word_index] = weight_parameter[word_index]
    return word_embedding_dic, idx2emb_dict

#获取Kmer字典
def getKmerVocab(word_embeddings_dic):
    vocab = word_embeddings_dic.keys()
    return vocab, len(vocab)

def findDiff(seq1, seq2):
    len1 = len(seq1)
    len2 = len(seq2)
    if len1 < len2:
        temp = seq2
        seq2 = seq1
        seq1 = temp
        tem = len1
        len2 = len1
        len1 = tem
    same_kmer = 0
    for i in range(len2):
        if seq1[i] == seq2[i]:
            same_kmer = same_kmer + 1
    diff_kmer = len1 - same_kmer
    return same_kmer, diff_kmer

def diff_seq(seq1, seq2, virus1, virus2):

    diff_pos_1 = diff_dict[virus1]
    diff_pos_2 = diff_dict[virus2]
    pad_seq1_list = "#"*diff_pos_1 + seq1
    pad_seq2_list = "*"*diff_pos_2 + seq2
    min_pos = min(diff_pos_1, diff_pos_2)

    min_len = min(len(pad_seq1_list), len(pad_seq2_list))

    length = min_len - min_pos
    same_mer = 0
    for i in range(min_pos, min_len):
        if pad_seq1_list[i] == pad_seq2_list[i]:
            same_mer = same_mer + 1

    return float(same_mer/length)

def virus_year(virus_year_data_path):
    csvFile = open(virus_year_data_path, "r")
    reader = csv.reader(csvFile)
    # 建立空字典
    virus_year = {}
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        virus_year[item[0].strip("\n")] = int(item[1])
    return virus_year

def getSequenceInputs(kmer_data):
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    virus_year_dict = virus_year("../virus_year.csv")

    n_index = 0
    for triple in kmer_data:
        seq1 = triple[0].upper()
        seq2 = triple[1].upper()
        virus1 = triple[3]
        virus2 = triple[4].replace("\n","")
        # print(len(seq1))
        seq1, seq2 = diff_seq(seq1, seq2, virus1, virus2)
        if virus_year_dict[virus1] < 1996 and virus_year_dict[virus2] < 1996:
            # if n_index%5 != 0 :
            train_data.append([seq1, seq2])
            train_labels.append(float(triple[2].replace('\n', '')))
        else:
            test_data.append([seq1, seq2])
            test_labels.append(float(triple[2].replace('\n', '')))
        n_index = n_index + 1
    print(len(train_data))
    print(len(test_data))
    print("dataset_distribution")
    return train_data, train_labels, test_data, test_labels

def Process_pair( data):
    random.seed(42)
    train_data = []
    test_data = []
    test_labels = []
    train_labels = []
    train_re_virus_name = []
    train_te_virus_name = []
    virus_year_dict = virus_year("../virus_year.csv")

    for sequence in data[1:]:
        seq1 = sequence[1].replace(" ", "")
        seq2 = sequence[3].replace(" ", "")
        train_labels.append(sequence[5])
        train_re_virus_name.append(sequence[0].strip())
        train_te_virus_name.append(sequence[2].strip())
        if virus_year_dict[sequence[0].strip()] < 1996 and virus_year_dict[sequence[2].strip()] < 1996:
            train_data.append([[seq1,seq2, sequence[0].strip(), sequence[2].strip()], sequence[5]])
        else:
            test_data.append([[seq1,seq2, sequence[0].strip(), sequence[2].strip()], sequence[5]])

    random.shuffle(train_data)
    random.shuffle(test_data)
    train_seq = [d[0] for d in train_data]
    train_labels = [d[1] for d in train_data]
    test_seq = [d[0] for d in test_data]
    test_labels = [d[1] for d in test_data]
    return train_seq, train_labels, test_seq, test_labels

if __name__ == '__main__':
    print('Start read data')
    features = []
    labels = []
    time_1 = time.time()
    # #获取kmer_pair
    # #计算相似性（根据embedding 相似性），训练一个阈值，利用这个阈值作为是否产生相似抗体的依据
    path = "../data_constructor/dataset-4labels.xlsx"
    r = ReadExcel(path)
    # 获取到所有基因序列与最长基因序列的开始位置的距离字典
    diff_dict = getDiffDic()
    table_data, headers = r.readExcel(0)
    train_data, train_label, test_data, test_label = Process_pair(table_data)

    train_features = []
    train_labels = []
    for i in range(len(train_data)):
        similarity = diff_seq(train_data[i][0], train_data[i][1], train_data[i][2], train_data[i][3])
        train_features.append([similarity])
        train_labels.append(train_label[i])

    test_features = []
    test_labels = []
    for i in range(len(test_data)):
        similarity = diff_seq(test_data[i][0], test_data[i][1], test_data[i][2], test_data[i][3])
        test_features.append([similarity])
        test_labels.append(test_label[i])

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')
    print(train_features)
    print('Start training')
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')

    dtree_model = DecisionTreeClassifier(max_depth=5).fit(train_features, train_labels)
    dtree_predictions = dtree_model.predict(test_features)
    print(dtree_predictions)

    tree_predicts = [[i] for i in dtree_predictions]
    print(classification_report(test_labels, tree_predicts, digits=4))