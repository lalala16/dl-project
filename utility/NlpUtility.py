#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding:utf-8
import numpy as np
import jieba



def serperate_text(text_contend='', text2sentence=False):
    text_list = jieba.lcut(text_contend)
    stop_sentence = {u'。', u'！', u'？', u';'}
    if not text2sentence:

        return text_list
    else:
        result = []
        line = []
        for count, word in enumerate(text_list):
            if word in stop_sentence:
                result.append(line)
                line = []
            else:
                line.append(word)
        return result





if __name__ == '__main__':
    print(serperate_text('我在开会。youarehansome！你在干吗？我是谁;哈哈哈。', text2sentence=True))