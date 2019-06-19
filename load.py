"""
Python program to extract data from the csv file and load them into the python program.
The data set is divided as follows:
train_x, train_y: m = 29999
valid_x, valid_y: m = 10567

no. of features: n = 3

These are the data-sets which we generated by extracting the coding/non-coding regions of the prokaryotic genome.
(DNA.zip which was shared with all of us)

Calling this function from this snippet would extract all the required data
and store them in python variables.
"""

import csv
import numpy as np 
import tensorflow as tf 

def load_data():
	train_x = []; train_y = []
	valid_x = []; valid_y = []

	with open('train_input.csv','r') as csvfile:
		flurg = csv.reader(csvfile)
		for row in flurg:
			train_x.append([float(num) for num in row])

	with open('train_output.csv', 'r') as csvfile:
		flurg = csv.reader(csvfile)
		for row in flurg:
			train_y.append([int(num) for num in row])

	with open('test_input.csv', 'r') as csvfile:
		flurg = csv.reader(csvfile)
		for row in flurg:
			valid_x.append([float(num) for num in row])

	with open('test_output.csv', 'r') as csvfile:
		flurg = csv.reader(csvfile)
		for row in flurg:
			valid_y.append([int(num) for num in row])

	return train_x, train_y, valid_x, valid_y		

if __name__ == '__main__':
	a,b,c,d = load_data()
	print (d)