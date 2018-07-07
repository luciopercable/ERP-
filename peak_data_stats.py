#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:	   peak_data_stats.py
# Purpose:   Erp statistics
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: 23.03.18
# ----------------------------------------------------------------------------


import numpy as np
import pandas as pd

from scipy import stats
import sys, itertools, pickle

def wilcox(data):
	output = []
	for comb in itertools.combinations(data.keys(), 2):
		output.append([str(comb),  [ np.mean(data[a]) for a in comb ], stats.wilcoxon(*[ data[a] for a in comb ])])
	return output

def student(data):
	output = []
	for comb in itertools.combinations(data.keys(), 2):
		output.append([str(comb),  [ np.mean(data[a]) for a in comb ], stats.ttest_rel(*[ data[a] for a in comb ])])
	return output

def corr(data):
	channels = [a for a in data.keys() if a != 'File']
	for channel1 in channels:
		for channel2 in channels:
			if channel1 != channel2:
				for regs in data[channel1].keys():
					# print data[channel1][reg]
					st = stats.kendalltau(data[channel1][reg], data[channel2][reg])
					if st.pvalue < 0.05/(len(channels)**2):
						print channel1, channel2, st

def shapiro(data):
	ret = {}
	for var in data.keys():
		ret[var] = True if stats.shapiro(data[var])[1] >= 0.05 else False
	return ret


def pairwise (total_data, stats = False, norm = False):
	import xlsxwriter #1
	workbook = xlsxwriter.Workbook('my_excel_new.xlsx')
	big_diff = workbook.add_format(properties={'font_color' : 'red'})
	middle_diff = workbook.add_format(properties={'font_color': 'yellow'})
	small_diff = workbook.add_format(properties={'font_color': 'green'})
	row_num_1 = 0
	row_num_2 = 0
	row_num_3 = 0
	row_num_4 = 0


	row_num = 0

	worksheet_static = workbook.add_worksheet('static')
	worksheet_pvalue = workbook.add_worksheet('pvalue')
	m_save_good_for_analis = []
	m_save_good_for_analis_channel = []
	m_all = [a for a in total_data.keys() if a != 'File']
	for channel in [a for a in total_data.keys() if a != 'File']:
		if channel.find('p3a') != -1:
			cur_step = 0
			row_num = row_num_1
		if channel.find('p3i') != -1:
			cur_step = 6
			row_num = row_num_2
		if channel.find('n1a') != -1:
			cur_step = 12
			row_num = row_num_3
		if channel.find('n1i') != -1:
			cur_step = 18
			row_num = row_num_4
		worksheet_pvalue.write(row_num, cur_step, channel) #
		worksheet_static.write(row_num, cur_step, channel)  #

		if norm:
			normality = shapiro(total_data[channel])
			# print channel, normality
		elif stats:
			if stats == 'student':
				pairwise_stats = student(total_data[channel])
			elif stats == 'wilcoxon':
				pairwise_stats = wilcox(total_data[channel])

			dict = {'Facesnoise' : {'Facesnoise' : {'static' : 0, 'pvalue' : 0}, 'Noise' : {'static' : 0, 'pvalue' : 0}, 'Letters' : {'static' : 0, 'pvalue' : 0}, 'Faces' : {'static' : 0, 'pvalue' : 0}},
					'Noise' : {'Facesnoise' : {'static' : 0, 'pvalue' : 0}, 'Noise' : {'static' : 0, 'pvalue' : 0}, 'Letters' : {'static' : 0, 'pvalue' : 0}, 'Faces' : {'static' : 0, 'pvalue' : 0}},
					'Letters' : {'Facesnoise' : {'static' : 0, 'pvalue' : 0}, 'Noise' : {'static' : 0, 'pvalue' : 0}, 'Letters' : {'static' : 0, 'pvalue' : 0}, 'Faces' : {'static' : 0, 'pvalue' : 0}},
					'Faces' : {'Facesnoise' : {'static' : 0, 'pvalue' : 0}, 'Noise' : {'static' : 0, 'pvalue' : 0}, 'Letters' : {'static' : 0, 'pvalue' : 0}, 'Faces' : {'static' : 0, 'pvalue' : 0}}}
			for ps in pairwise_stats:
				str = ps[0].replace('\\', '')
				str = str.replace('(', '')
				str = str.replace(')', '')
				str = str.replace("'", "")
				str = str.replace(" ", "")
				p1 = str.split(',')[0]
				p2 = str.split(',')[1]
				dict[p1][p2]['static'] = ps[2][0]
				dict[p1][p2]['pvalue'] = ps[2][1]
				dict[p2][p1]['static'] = ps[2][0]
				dict[p2][p1]['pvalue'] = ps[2][1]
				dict[p1][p1]['pvalue'] = 1
				dict[p2][p2]['pvalue'] = 1
			row_num += 1
			names = dict.keys()
			for num,i in enumerate(dict.keys()):
				worksheet_pvalue.write(row_num, num + 1 + cur_step, i)
				worksheet_pvalue.write(row_num + num + 1 , 0 + cur_step, i)
				worksheet_static.write(row_num, num + 1  + cur_step, i)
				worksheet_static.write(row_num + num + 1, 0  + cur_step, i)
			for i in range(0, len(names)):
				for j in range(0, len(names)):
					value = dict[names[i]][names[j]]['pvalue']
					value_static = dict[names[i]][names[j]]['static']
					if value < 0.05 / 6:
						if i > j:
							m_save_good_for_analis.append([channel, names[i], names[j], value])
							if channel not in m_save_good_for_analis_channel:
								m_save_good_for_analis_channel.append(channel)
						worksheet_pvalue.write(row_num + 1 + i, 1 + j + cur_step, value, big_diff)
						worksheet_static.write(row_num + 1 + i, 1 + j + cur_step, value_static, big_diff)
					else:
						worksheet_pvalue.write(row_num + 1 + i, 1 + j + cur_step, value, small_diff)
						worksheet_static.write(row_num + 1 + i, 1 + j + cur_step, value_static, small_diff)
			row_num += 6
			if channel.find('p3a') != -1:
				row_num_1 = row_num
			if channel.find('p3i') != -1:
				row_num_2 = row_num
			if channel.find('n1a') != -1:
				row_num_3 = row_num
			if channel.find('n1i') != -1:
				row_num_4 = row_num
	import matplotlib.pyplot as plt
	import matplotlib.lines as mlines
	import math
	fig = plt.figure()
	# Create an axes instance
	max_x = 3
	# Create the boxplot
	fig = plt.figure(figsize=(8, 6), dpi=120)
	#fig.set_size_inches(18.5, 30.5, forward=True) #first means x, second means y
	m_bp = []
	m_subplots = None
	m_save_good_for_analis_channel = m_all
	names = ['Letters', 'Faces', 'Noise', 'Facesnoise']
	p3a_f = open('p3a_median.txt', 'w')
	p3i_f = open('p3i_median.txt', 'w')
	n1a_f = open('n1a_median.txt', 'w')
	n1i_f = open('n1i_median.txt', 'w')

	for num_subplot, cur_subplot in enumerate(m_save_good_for_analis_channel):
		print m_save_good_for_analis_channel
		print(cur_subplot)
		#m_subplots.append(fig.add_subplot(math.ceil(len(m_save_good_for_analis_channel) * 1.0 / max_x), max_x, num_subplot + 1))
		m_subplots = fig.add_subplot(111)
		data = []
		#for all in names:
		data.append(total_data[cur_subplot]['Letters'])
		data.append(total_data[cur_subplot]['Faces'])
		data.append(total_data[cur_subplot]['Noise'])
		data.append(total_data[cur_subplot]['Facesnoise'])

		median_prop = {'color' : 'red', 'linewidth' : '3'}
		m_bp = m_subplots.boxplot(data, medianprops = median_prop, patch_artist=True)
		#m_subplots[-1].set_ylabel(u'Значение')
		#m_subplots[-1].set_xlabel(u'Название режима')
		#m_subplots[-1].set_title(cur_subplot)
		#m_subplots[-1].set_xticklabels(names)
		m_subplots.set_ylabel(u'Значение')
		m_subplots.set_xlabel(u'Название режима')
		m_subplots.set_title(cur_subplot)
		#print m_subplots.get_ylim()[1]
		all_len = m_subplots.get_ylim()[1] - m_subplots.get_ylim()[0]
		m_subplots.set_ylim([m_subplots.get_ylim()[0], m_subplots.get_ylim()[1] + all_len / 4.0])
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
		for cur_not_cor in m_save_good_for_analis:
			if cur_subplot == cur_not_cor[0]:
				ind_1 = 0
				ind_2 = 0
				for num,cur_name in enumerate(names):
					if cur_name == cur_not_cor[1]:
						ind_1 = num
					if cur_name == cur_not_cor[2]:
						ind_2 = num
				start = min(pos[ind_1], pos[ind_2])
				end = max(pos[ind_1], pos[ind_2])
				print (start, end)
				l = mlines.Line2D([start, end], [y_pos + num_cur_cor * all_len / 25, y_pos + num_cur_cor * all_len / 25])
				m_subplots.add_line(l)
				num_cur_cor += 1
		#label_names = [u'Лица на фоне шума',  u'Шум', u'Буквы', u'Лица']
		label_names = [u'Буквы', u'Лица', u'Шум', u'Лица на фоне шума']
		m_subplots.set_xticklabels(label_names)
		for box in m_bp['boxes']:
			box.set(color = 'black', linewidth = 3)
			box.set(facecolor = '#acacaf')
		#m_subplots.legend(('1', '2', '3', '4'))
		f = open('median for all','a')
		f.write(cur_subplot + '\n')
		print ("Medians")
		for num, all in enumerate(names):
			print ("Median for ", all, ' is ', m_bp['medians'][num]._y[0])
			num = "%s" % m_bp['medians'][num]._y[0]
			if cur_subplot.find('p3a') != -1:
				p3a_f.write(cur_subplot + ' ' + all + ' ' + num + '\n')
			if cur_subplot.find('p3i') != -1:
				p3i_f.write(cur_subplot + ' ' + all + ' ' + num + '\n')
			if cur_subplot.find('n1a') != -1:
				n1a_f.write(cur_subplot + ' ' + all + ' ' + num + '\n')
			if cur_subplot.find('n1i') != -1:
				n1i_f.write(cur_subplot + ' ' + all + ' ' + num + '\n')
			f.write("Median for " +  all + ' is ' + num + '\n')
		f.close()
		num = "%s" % num_subplot
		#fig.savefig('plots/fig_' + cur_subplot  + '.png', bbox_inches='tight')
		m_subplots.clear()
		fig.clear()
	#plt.show()
	p3a_f.close()
	p3i_f.close()
	n1a_f.close()
	n1i_f.close()
	workbook.close()

				#if ps[2].pvalue < 0.05/6:
				# if 1:
				#	print channel, ps
				#	worksheet

if __name__ == '__main__':
	with open('peaks_av.pickle', 'rb') as file_obj:
		total_data = pickle.load(file_obj)
	eye_data = False
	if eye_data:
		flist =  [a for a in total_data['File']]#.split('.')[0]]
		eye_data = pd.read_csv('eye_measures_data.csv')
		eye_data['User'] = [int(a.split('-')[0]) for a in eye_data['File']]
		eye_data['Reg'] = [a.split('-')[1].split('_')[0].capitalize() for a in eye_data['File']]
		eye_data =  eye_data.sort_values(by=['User'])
		for col in eye_data.columns:
			if col not in ['File', 'User', 'Reg', 'Unnamed: 0']:
				# print eye_data['File'][eye_data['Reg'] == 'Facesnoise']
				dic = {}
				for reg in set(eye_data['Reg']):
					dic[reg] =  list(eye_data[col][eye_data['Reg'] == reg]) 
				total_data[col] = dic

	pairwise(total_data, stats = 'wilcoxon')
	# corr(total_data)

