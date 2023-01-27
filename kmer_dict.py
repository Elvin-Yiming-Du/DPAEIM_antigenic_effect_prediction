#Kmer_Dict用于构造k-mer的字典
import torch
import numpy as np
class Kmer_Dict:
    def __init__(self):
        self.word_embeddings_dic = {}
        self.idx2emb_dict = {}
        self.kmer2idx = {}
        self.vocab = []
        self.length = 0
    def create_embedding(self, vocab_path, weight_parameter):
        with open(vocab_path, "r", encoding="utf-8") as vocab:
            vocabulary = vocab.readlines()
            words = []
            for word in vocabulary:
                words.append(word.replace("\n", ""))

        for word_index in range(len(words)):
            self.word_embeddings_dic[words[word_index]] = weight_parameter[word_index]
            self.idx2emb_dict[word_index] = weight_parameter[word_index]
            self.kmer2idx[words[word_index]] = word_index
        self.vocab = words
        self.length = len(words)
    def add_new_kmer(self, kmer):
        new_kmer_embedding = torch.ones_like(self.word_embeddings_dic['[PAD]'])
        self.word_embeddings_dic[kmer] = new_kmer_embedding
        self.vocab.append(kmer)
        self.idx2emb_dict[self.length] = kmer
        self.kmer2idx[kmer] = self.length
        self.length = self.length + 1
        return new_kmer_embedding

    def getSeqEmbedding(self, seq):
        sentence_embedding = torch.ones_like(self.word_embeddings_dic["[PAD]"])
        for kmer in seq.split(' '):
            if kmer in self.word_embeddings_dic.keys():
                sentence_embedding = self.word_embeddings_dic[kmer] + sentence_embedding
            else:
                sentence_embedding = sentence_embedding + self.add_new_kmer(kmer)

        sentence_embedding = 1 / len(seq.split(' ')) * sentence_embedding
        return sentence_embedding

    def getSeqEmbeddingMatrix(self, seq, max_length):
        sentence_matrix = np.zeros(max_length)

        sentence_embedding = torch.ones_like(self.word_embeddings_dic["[PAD]"])
        kmers = seq.split(' ')
        for index, value in enumerate(kmers):
            if  value in self.word_embeddings_dic.keys():
                sentence_matrix[index] =self.word_embeddings_dic[value]
            else:
                sentence_matrix[index] = self.add_new_kmer(value)

        sentence_matrix[max_length-len(kmers): ] = sentence_embedding

        return sentence_matrix


    def get_word_embeddings_dic(self):
        return  self.word_embeddings_dic

    def get_vocab(self):
        return self.vocab

    def get_kmer2idx(self):
        return self.kmer2idx

    def get_idx2emb(self):
        return self.idx2emb_dict