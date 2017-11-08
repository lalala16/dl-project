#  Author: Yang Kai
#  Date: 11/2/2017
#
#
# *************************************** #

import pickle
import numpy as np
from utility.NlpUtility import serperate_text


class Vocabulary(object):
    def __init__(self, original_path=""):
        self.word2index = {}
        self.index2word = {}
        self.sort_count = {}
        self.maxlength = -1
        self.path = original_path

    def get_vocabulary(self, text_contend=[], remove_stopwords=[], topnum=-1, save=False):
        wordcount = {}
        for line in text_contend:
            if len(line) > self.maxlength:
                self.maxlength = len(line)
            for word in np.array(line).flatten():
                if word not in remove_stopwords:
                    if word in wordcount:
                        wordcount[word] = wordcount[word] + 1
                    else:
                        wordcount[word] = 1
        self.sort_count = sorted(wordcount.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

        index = 1

        for count_tuple in self.sort_count:
            word = count_tuple[0]
            self.word2index[word] = index
            self.index2word[index] = word
            index = index+1
            if topnum > 0 and index > topnum:
                break

        print "*************"
        print 'maxlenth: ', self.maxlength
        print 'wordset: ', len(self.sort_count)
        print 'w2i:', len(self.word2index)
        print 'i2w', len(self.index2word)
        if save == True:
            self.save_wordcount()


    def save_dict(self):
        file = open(self.path+'dictionary.pkl', 'wb')
        pickle.dump(self.word2index, file)
        pickle.dump(self.index2word, file)

    def save_wordcount(self):
        f1 = open(self.path+'count.txt','w+')
        for key in self.sort_count:
            f1.write(str(key[0])+":"+str(key[1])+"\n")
        f1.close()

    def get_onehot_vec(self, new_contend=[]):
        vec = [0] * len(self.index2word)
        for word in new_contend:
            vec[self.word2index[word]] += 1
        return vec

    def get_feature_list(self, new_contend=[], have_sentence=False):
        feature_list = []
        if not have_sentence:
            for word in new_contend:
                feature_list.append(self.word2index[word])
        else:
            for sentence in new_contend:
                sentence_feature = []
                for word in sentence:
                    sentence_feature.append(self.word2index[word])
                feature_list.append(sentence_feature)
        return feature_list




if __name__ == '__main__':

    original_path = '/Users/yangkai/Desktop/hackathon/sentiment-analysis-datasets/tweets-datasets/'
    voc = Vocabulary(original_path)
    # Document2VecUtility.get_wordlist(original_path + 'train.tsv', 3)
    filelist = ['training.1600000.processed.noemoticon.csv']
    # filelist = ['train.tsv','test.tsv','dev.tsv']
    voc.get_vocabulary(filelist, column_num=5)
    # voc.save_dict()
    voc.save_wordcount()



