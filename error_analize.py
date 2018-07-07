#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

good_word = '@neuroscience!'
for num,main_files in enumerate(os.listdir('valid')):
    print num
    for num1, cur_files1 in enumerate(os.listdir('valid/' + main_files)):
        if cur_files1.find('.txt') != -1:
            continue
        for cur_files in os.listdir('valid/' + main_files + '/' + cur_files1):
            if cur_files.find('.txt') != -1:
                f = open('valid/' + main_files + '/' + cur_files1 + '/' + cur_files)
                count_error = 0
                for line in f:
                    for i in range(len(line)):
                        if line[i] != good_word[i]:
                            count_error += 1
                print 'User : %s Mode: %s Count error: %d' % (main_files, cur_files1, count_error)


import numpy as np
import pandas as pd

from scipy import stats
import sys, itertools, pickle

def student(data):
	output = []
	for comb in itertools.combinations(data.keys(), 2):
		output.append([str(comb),  [ np.mean(data[a]) for a in comb ], stats.ttest_rel(*[ data[a] for a in comb ])])
	return output

def wilcox(data):
	output = []
	for comb in itertools.combinations(data.keys(), 2):
		output.append([str(comb),  [ np.mean(data[a]) for a in comb ], stats.wilcoxon(*[ data[a] for a in comb ])])
	return output

acc_faces = [0.785714286,1,1,0.785714286,1,0.857142857,1,1,1,1,1,1,1,1,1,0.928571429,1,0.785714286,0.928571429,1,1,1,1,]
acc_facesnoise = [1,1,1,0.928571429,1,0.928571429,1,1,0.785714286,0.928571429,0.5,0.928571429,0.857142857,0.928571429,1,0.928571429,1, 0.5, 0.857142857,1,1,1,1,]
acc_letters = [0.928571429,1,1,0.857142857,1,0.642857143,0.785714286,0.857142857,0.428571429,0.928571429,0.857142857,0.928571429,0.785714286,0.928571429,0.928571429,1,1,0.928571429,0.928571429,0.785714286,0.928571429,0.857142857,0.928571429]
acc_noise = [0.857142857,1,1,0.785714286,1,1,1,1,0.785714286,1,1,1,1,1,0.928571429,1,1,1,1,1,1,1,1,]
dict_acc = {'Faces' : acc_faces, 'Facesnoise' : acc_facesnoise, 'Letters' : acc_letters, 'Noise' : acc_noise}
print wilcox(dict_acc)
acc_facesnoise.sort()
print acc_facesnoise
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

data = [acc_letters, acc_faces, acc_noise, acc_facesnoise]
fig = plt.figure(figsize=(8, 6), dpi=120)
median_prop = {'color' : 'red', 'linewidth' : '3'}
m_subplots = fig.add_subplot(111)
m_bp = m_subplots.boxplot(data, medianprops = median_prop, patch_artist=True)
m_subplots.set_ylabel(u'Точность', fontsize = 16)
#m_subplots.set_xlabel(u'Название режима')
m_subplots.set_title(u'Зависимость точности от режима стимуляции', fontsize = 16, fontweight = 'bold')
#print m_subplots.get_ylim()[1]
all_len = m_subplots.get_ylim()[1] - m_subplots.get_ylim()[0]
#m_subplots.set_ylim([m_subplots.get_ylim()[0], m_subplots.get_ylim()[1] + all_len / 4.0])
#m_subplots.annotate("$C_{3}H_{8}$", xy=(0.9, 0.9), xycoords='axes fraction', fontsize=14)
pos = []
y_pos = 0
for i in range(0, 4, 1):
    pos.append((m_bp['caps'][i * 2]._x[0] + m_bp['caps'][i * 2]._x[1]) * 1.0 / 2)
    y_pos = max(m_bp['caps'][i * 2]._y[0], y_pos)
    y_pos = max(m_bp['caps'][i * 2]._y[1], y_pos)
    y_pos = max(m_bp['caps'][i * 2 + 1]._y[0], y_pos)
    y_pos = max(m_bp['caps'][i * 2 + 1]._y[1], y_pos)
num_cur_cor = 0
label_names = [u'Буквы', u'Лица', u'Шум', u'Лица на фоне\nшума']
#label_names = [u'Letters', u'Face', u'Noise', u'Facenouse']
m_subplots.set_xticklabels(label_names, fontsize = 16)
m_subplots.set_yticklabels(['0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize = 16)
for box in m_bp['boxes']:
    box.set(color = 'black', linewidth = 3)
    box.set(facecolor = '#acacaf')
print ("Medians")
for num, all in enumerate(label_names):
    print ("Median for ", all, ' is ', m_bp['medians'][num]._y[0])
    num = "%s" % m_bp['medians'][num]._y[0]

pvalue=0.0025117556612396044
y_pos = m_subplots.get_ylim()[1]
print y_pos
l = mlines.Line2D([1.0, 3.0], [y_pos + all_len / (16 * 8), y_pos + all_len / (16 * 8)], color='black')
m_subplots.add_line(l)
m_subplots.annotate("*", xy=(2 - 0.07, - all_len / (16 * 8) + y_pos), fontsize=24, fontweight = 'bold')
m_subplots.set_ylim([m_subplots.get_ylim()[0], m_subplots.get_ylim()[1] + all_len / 16.0])
plt.show()
#fig.savefig('plots/fig_' + cur_subplot  + '.png', bbox_inches='tight')
