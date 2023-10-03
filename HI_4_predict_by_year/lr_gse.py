import csv
import json
import torch
import numpy as np
import xlrd
import random
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from xlwings import xrange
from sklearn.tree import DecisionTreeClassifier

from kmer_dict import Kmer_Dict
from src.transformers import BertModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Perceptron(object):
    def __init__(self):
        self.learning_step = 0.0001
        self.max_iteration = 50

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
    virus_year_dict = virus_year("../virus_year.csv")
    positive = 0
    negative = 0
    for triple in kmer_data:
        if int(triple[2].replace('\n','')) == 1:
            positive = positive + 1
        else:
            negative = negative + 1

    n_index = 0
    for triple in kmer_data:
        seq1 = triple[0].upper()
        seq2 = triple[1].upper()
        virus1 = triple[3]
        virus2 = triple[4].replace("\n","")

        seq1_emb = kmer_dict.getSeqEmbedding(seq1)
        seq2_emb = kmer_dict.getSeqEmbedding(seq2)
        if virus_year_dict[virus1] < 1996 and virus_year_dict[virus2] < 1996:
            train_data.append([[seq1_emb.numpy(), seq2_emb.numpy()], float(triple[2].replace('\n',''))])
        else:
            test_data.append([[seq1_emb.numpy(), seq2_emb.numpy()], float(triple[2].replace('\n',''))])
        n_index = n_index + 1

    random.shuffle(train_data)
    random.shuffle(test_data)
    train_seq = [d[0] for d in train_data]
    train_labels = [d[1] for d in train_data]
    test_seq = [d[0] for d in test_data]
    test_labels = [d[1] for d in test_data]
    print(len(train_data))
    print(len(test_data))
    print("dataset_distribution")
    print(positive)
    print(negative)
    return train_seq, train_labels, test_seq, test_labels

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
    model = BertModel.from_pretrained("../dnabert")
    embeddings = model.get_input_embeddings()
    weight_parameter = embeddings.weight.data
    vocab_path = "../dnabert/vocab.txt"
    data_path = "../data_constructor/VHID_4_degree.tsv"

    # 获取embedding字典word_embeddings_dic，举例，key为AAATTC，value为768维的tensor;idx2emb_dic,key为AAATTC，value为768维的tensor
    kmer_dict = Kmer_Dict()
    kmer_dict.create_embedding(vocab_path, weight_parameter)

    # 获取数据data [[seq1,seq2,label],...],seq1指的是ATTCCG AATCCT ...,seq2指的是ATTCCT ATTCCG
    kmer_pairs = getKmer(data_path)
    print("data总量：")
    print(len(kmer_pairs))
    # 获取到所有基因序列与最长基因序列的开始位置的距离字典
    diff_dict = getDiffDic()
    word_embeddings_dic = {}

    train_data, train_label, test_data, test_label = getSequenceInputs(kmer_pairs)

    train_features = []
    train_labels = []
    for i in range(len(train_data)):
        similarity = cosine_similarity(np.array([train_data[i][0]]), np.array([train_data[i][1]]))
        train_features.append(similarity[0])
        train_labels.append(train_label[i])

    test_features = []
    test_labels = []
    for i in range(len(test_data)):
        similarity = cosine_similarity(np.array([train_data[i][0]]), np.array([train_data[i][1]]))
        test_features.append(similarity[0])
        test_labels.append(test_label[i])

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')
    print(train_features)
    print('Start training')

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', max_iter=10000)
    # p = Perceptron()
    lr.fit(train_features, train_labels)
    train_score = accuracy_score(train_labels, lr.predict(train_features))
    test_predict = lr.predict(test_features)
    test_labels = [[j] for j in test_labels]
    print('测试数据指标:\n', classification_report(test_labels, test_predict, digits=4))

