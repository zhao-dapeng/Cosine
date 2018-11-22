#encoding:utf8
import re
import time
import math
import jieba
import pymysql
import pandas as pd

from math import sqrt
from functools import reduce
from operator import itemgetter

from article_similarity.tfidf import TfIdf
from article_similarity.naivebayesianpre import NaiveBayesian


class Cosine(object):

    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2

    def jieba(self,x_text):
        jieba.suggest_freq("区块链", True)#调节单个词语的词频，使其能或不能被分出来
        segs = jieba.cut(x_text, cut_all=False, HMM=False)
        segs = ('/'.join(segs)).split('/')#将分好的词使用/连接去掉首尾的/
        segs=filter(lambda x: len(x)>1,segs)#过滤出单个词
        segs=filter(lambda x:x not in stopwords,segs)#过滤出停用词
        list_tags=[]
        for tags in segs:
            if re.match('[0-9a-z\.A-Z%]+',tags):
                pass
            else:
                list_tags.append(tags)
        return list_tags

    def gettfidf(self,y_text):
        keyword1 = Tf.one_tfidf(y_text,20)
        return keyword1

    def setofwords2vec(self,vocablist, inputSet):
        returnVec = [0]*len(vocablist)#创建一个其中所含元素都为0的向量
        for word in inputSet:#遍历文档中所有的单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
            if word in vocablist:
                returnVec[vocablist.index(word)] += 1
        return returnVec

    def similar(self):
        j1 = self.jieba(self.text1)
        j2 = self.jieba(self.text2)
        strj1 = ' '.join(j1)
        strj2 = ' '.join(j2)
        train1 = self.gettfidf(strj1)
        category1 = naiveBayesian.Predict(train1)
        train2 = self.gettfidf(strj2)
        category2 = naiveBayesian.Predict(train2)
        if category1 == category2:#判断是否是相同类别
            trainlist = list(set(train1+train2))
            res1 = self.setofwords2vec(trainlist,j1)
            res2 = self.setofwords2vec(trainlist,j2)
            #余弦定理
            cos1 = sqrt(reduce(lambda x,y:x+y,map(lambda x:x*x,[i for i in res1])))
            cos2 = sqrt(reduce(lambda x,y:x+y,map(lambda x:x*x,[i for i in res2])))
            sum = reduce(lambda x,y:x+y,map(lambda x1,x2:x1*x2,[x1 for x1 in res1],[x2 for x2 in res2]))
            res = (sum)/(cos1*cos2+1)
            return res
        else:
            return 0.0


if __name__ == '__main__':
    t1 = 'xxxxxxxxxxxxxxxxxxxxx'
    t2 = 'yyyyyyyyyyyyyyyyyyyyy'
    Tf=TfIdf()
    jieba.load_userdict('./CustomDictionary.txt')#加载自定义词典
    stopwords = pd.read_csv('./stopwords.txt', index_col=False, quoting=3, sep='\t', names=['stopword'],encoding='utf-8')
    stopwords = stopwords['stopword'].values
    naiveBayesian = NaiveBayesian()
    naiveBayesian.LoadModel(r'./NaiveBayesianModel2018426_echo2')
    cos = Cosine(t1,t2)
    cos.similar()
