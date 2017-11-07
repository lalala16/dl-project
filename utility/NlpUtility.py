#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding:utf-8
import numpy as np
import jieba



def serperate_text(text_contend=''):
    return jieba.lcut(text_contend)


if __name__ == '__main__':
    print(serperate_text('我在开会,youarehansome'))