# -*- coding : utf-8 -*-
#use coordinate gradient descent to solve LASSO formulation, which is convex
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import time
import pickle
import random
import os
import sys
import math



file_info = (
		'''
		Differential Coding
		Testing
		Author: MrZQ
		20181107

		v1.0.0.1112_alpha Update
				1. Basic framework coming from CoD
				2. Bug fix
				3. Appendix
						1. np.linalg.norm----> 范数

		v1.0.1.1113_alpha Update
				1. Bug fix
				2. Appendix
						1. main_operation_cod : 针对MSE的迭代
						2. main_operation_ista: 针对LASSO的迭代
						3. A : Dictionary
						4. X : Sparse coding
				3. Bug status:
						1. h_sign doesn`t work ---> fixed

		v1.0.2.1114_alpha Update
				1. Bug fixed
				2. 

		''')



#--------------------------------variables----------------------------------------------
# n p s --------> size parameters
n = 1000		# row size
p = 500		# column size
s = 100		# nonzero size
step_length = 10
#lmd = math.sqrt(2*n*math.log(p))  #lambda in LOSSA objective function
lmd = 0.5
#L = math.sqrt(2*n*math.log(p))
L = 0


# opt_x --------> dictionary
# b		--------> sample
A 		= []
opt_x	= []
b		= []








#----------------------------------Calculation function----------------------------------------------
'''
#compare a value with each element in a vector
# and replace the element with the max one, which is the value of origin
#return a new vector
def vec_max(value, vector):
	for i in range(len(vector)):
		vector[i] = max(value, vector[i])
	return vector


# gradient step
def gradient_descent(x_k, step_length,A,b):
	f_prime = np.dot(A.T, np.dot(A,x_k.T) - b) # derivertive of f_function
	print('f_prime '+ str(f_prime))
	y = x_k - step_length * f_prime
	return y
'''

def sign(y):
	if y > 0.0:
		return 1.0
	elif y < 0.0:
		return -1.0
	else:
		return 0.0


#proximal function
#lmd is the lambda value for convex closed function
# y is the [x_k - r_k* f'(x_k)]
#output prox_vec is p*1 dimensions
#def h_sign(lmd,step_length,y):
# threshold :lmd/L
def h_sign(y):
	new_y = y
	tmp = 0
	for i in range(len(y)):
		tmp = sign(y[i])
		#new_y[i] = tmp * max(0, abs(y[i]) - step_length*lmd)   # NOTE: here is step_length * lmd, use max instead vec_max function
		new_y[i] = tmp * max(0, abs(y[i]) - lmd/L)
		#new_y[i] = tmp * max(0, abs(y[i]) - lmd/2)
		#print('tmp is:' + str(tmp))
		#print('y[i] is:' + str(y[i]))
		#print('new_y[i] is:' + str(new_y[i]))
	
	#print('y is: \n' + str(y) )
	#print('new_y is: \n' + str(new_y) + '\n')

	#print('threshold is: ' + str(lmd/L))
	return new_y


def iteration_operator(A , x_k , b , i , trig):
	A_copy = np.copy(A)
	x_copy = np.copy(x_k)

	if trig == '1':
		y = np.dot((A[:,i]).T, b - np.dot(A_copy,x_copy.T))/np.dot((A[:,i]).T, A[:,i]) + x_copy[i]
	elif trig == '2':
		y = h_sign






#---------------------------------initilization------------------------------------------
def init_1():
	# when initialize a vector, had better to initialize it as row vector instead column vector
	time_1 = time.time()
	global A
	global opt_x
	global b
	global L


	A = np.random.normal(0,1,(n,p))		#0-1高斯分布

	L = float(int(np.linalg.eig(np.dot(A.T , A))[0][0]) + 1)
	print('L is: ' + str(L))


	opt_x = np.array([0.0]*p)				# optimum x, used for generating b, and try to get optimum x using b and A

	random_index_list = random.sample(range(p), s)
	for i in random_index_list:
		#opt_x[i] = np.random.normal(0,10)
		opt_x[i] = np.random.normal(0,1)


	e   = np.random.normal(0,1,n)
	b   = np.dot(A,opt_x.T) + e.T

	time_2 = time.time()
	#print('initialize time: ' + str(time_2 - time_1))




	'''
	print('random_index_list is: ' + str(random_index_list))
	print ('b shape is : ' + str(b.shape))
	print ('Dictionary shape is: '+ str(A.shape))
	
	print ('opt_x shape is: '+ str(opt_x.shape))
	
	'''
	print ('threhold is: \n' + str(lmd/2))
	print ('Dictionary A is: \n'+ str(A))
	print ('Origin opt_x is: \n'+ str(opt_x))
	print ('Origin b is: \n'+ str(b) + '\n\n')
	





#----------------------------------model function-----------------------------------------
# primary function f(x)_1 = (||Ax - b||^2) /2
# primary function f(x)_2 = (||Ax - b||^2) /2 + lmd||x||1
def f_func(B,x,b , trig):
	if trig == '1':
		f_value_1 = math.pow(np.linalg.norm(np.dot(B,x.T) - b),2) / 2
		return f_value_1
	elif trig == '2':
		f_value_2 = (math.pow(np.linalg.norm(np.dot(B,x.T) - b),2) / 2)  +  (lmd * np.linalg.norm(x,1))
		return f_value_2

#Proximal Operator
#Second-order Taylor expansion + Regularization
def m_func(B, x_n, b, s, x , lmd ,trig2):
	part_1 = f_func(B,x_n,b , '1')
	part_2 = np.dot( (np.dot(B.T, np.dot(B,x_n.T) - b)).T, (x-x_n).T )
	part_3 = math.pow(np.linalg.norm(x-x_n),2) / (2*s)
	part_4 = lmd * np.linalg.norm(x,1)
	part_5 = math.pow(np.linalg.norm(x-x_n),2) * s / 2


	if trig2 == '1':
		m_value_1 = part_1 + part_2 + part_3
		return m_value_1

	#--------------------------------debug start------------------------------------
	# using part 5
	elif trig2 == '2':
		m_value_2 = part_1 + part_2 + part_5 + part_4
		return m_value_2



#objective function
# 这个lmb和1范数根本没用上，优化的就是一个LASSO
# 这个obj_func未使用
# 优化的还是f_func 与 m_func 之间
def obj_func(B,x,b,lmd):
	#f_value = math.pow(np.linalg.norm(np.dot(A,x.T) - b),2) / 2
	f_value = f_func(B,x,b , '1')
	g_value = lmd * np.linalg.norm(x,1)
	#value = f_value + g_value
	value = f_value

	return value


#----------------------------------main operation-----------------------------------------

def main_operation_cod():
	# opt_x ------> sparse x vector with 20 non-zero elements
	# x_k   ------> preparing for iteration
	# b     ------> sample created by A x opt_x + e

	time_1 = time.time()


	trig = '1'
	x_k = np.array([0.0]*p)
	step_length = 10




	for j in range(100):
		#obj_value = obj_func(A,x_k,b,lmd)
		#print('obj_value is: ' + str(obj_value))

		#outfile.write(str(obj_value)+'\n')
		for i in range(p):
			while(1):
				#print 'x_k',x_k
				#y = gradient_descent(x_k, step_length, A, b)
				#print 'y',y
				#x_k_plus_1 = prox_operation(lmd,step_length, y)
				#print 'x_k_plus_1',x_k_plus_1
				A_copy = np.copy(A)
				x_copy = np.copy(x_k)
				#A_copy[:,i] = A_copy[:,i] * 0.0
				#x_copy[i] = 0
				#y = x_copy[i] - step_length* ( np.dot( (A[:,i]).T, b-np.dot(A_copy, x_copy.T) ) / np.dot((A[:,i]).T, A[:,i]))


				#print('A[:,i] is: ' + str(A[:,i]))
				#print('A[:,i] length is: ' + str(len(A[:,i])))



				#-------------------------------------------core algorithm-----------------------------------------
				y = np.dot((A[:,i]).T, b - np.dot(A_copy,x_copy.T))/np.dot((A[:,i]).T, A[:,i]) + x_copy[i]
				#y = iteration_operator(A , x_k , b ,i, trig)
				#print('y is: ' + str(y))

















				#new_x_i = prox_operation(lmd, step_length, y)

				x_k_plus_1 = np.copy(x_k)
				x_k_plus_1[i] = y
				#f_value = f_func_MSE(A,x_k_plus_1,b)
				f_value = f_func(A,x_k_plus_1,b,trig)
				m_value = m_func(A,x_k, b, step_length, x_k_plus_1 , lmd , '1')


				if f_value <= m_value:
					break

				step_length = step_length * 0.5
				#print ('********************************************')


			#---------------------------debug start----------------------------
			#break条件：1合理阈值 2固定迭代次数
			#if f_func_MSE(A,x_k,b) <= f_func_MSE(A,x_k_plus_1,b)+1e-8:
			if f_func(A,x_k,b , trig) <= f_func(A,x_k_plus_1,b , trig):
				break

			if i%500 == 0:
				print ('f_value is: ' + str(f_value))
				print ('m_value is: ' + str(m_value))


			else:
				x_k = x_k_plus_1

	time_2 = time.time()
	print( 'Problem: MSE\nMatrix size: ' + str(n) + ' x ' + str(p) + '\n' + 'Method: Coordinate descent\nRunning time: ' + str(time_2 - time_1) + '\n' )
	print( 'Final x_k is: ' + str(x_k) + '\n')
	#print( 'Final x_k_plus_1 is: ' + str(x_k_plus_1) + '\n')


def main_operation_ista():
	# y ----->  核心计算步骤
	# 通过理论推导出的公式来迭代
	# 比较当前与目标函数的值？
	# opt_x ------> sparse x vector with 20 non-zero elements
	# x_k   ------> preparing for iteration
	# b     ------> sample created by A x opt_x + e

	time_1 = time.time()
	trig = '2'
	x_k = np.array([0.0]*p)
	step_size = 10
	#L = math.sqrt(2*n*math.log(p))
	outfile = open('testing.log','w')


	for i in range(1000):
	#while(1):
		#print 'x_k',x_k
		#y = gradient_descent(x_k, step_length, A, b)
		#print 'y',y
		#x_k_plus_1 = prox_operation(lmd,step_length, y)
		#print 'x_k_plus_1',x_k_plus_1
		A_copy = np.copy(A)
		x_copy = np.copy(x_k)
		#A_copy[:,i] = A_copy[:,i] * 0.0
		#x_copy[i] = 0
		#y = x_copy[i] - step_length* ( np.dot( (A[:,i]).T, b-np.dot(A_copy, x_copy.T) ) / np.dot((A[:,i]).T, A[:,i]))

		#-------------------------------------------core algorithm-----------------------------------------
		#
		#y_tmp ------> nunpy.ndarray
		y_tmp = x_k - (np.dot(A.T , (np.dot(A,x_copy)-b)))/L
		#print('##### No. ' + str(i) + '########\n y_tmp is: \n' + str(y_tmp))

		y = h_sign( y_tmp  )
		
		#y = h_sign(  x_k - (np.dot(A.T , (np.dot(A,x_copy)-b)))/L   )

		log_tmp_1 = '\n##### No. ' + str(i) + ' ########\n x_k is: \n' + str(x_k)
		#print('\n##### No. ' + str(i) + ' ########\n x_k is: \n' + str(x_k))
		outfile.write(str(log_tmp_1)+'\n')
		
		
		
		#print('ista y_tmp is: \n' + str(y_tmp))
		#print('ista y is: \n' + str(y))



		#new_x_i = prox_operation(lmd, step_length, y)

		#x_k_plus_1 = np.copy(x_k)
		#x_k_plus_1[i] = y
		x_k_plus_1 = y
		#---------------------------debug start----------------------------
		#f_value = f_func_LASSO(A,x_k_plus_1,b , lmd)
		f_value = f_func(A,x_k_plus_1,b , trig)
		
		log_tmp_2 = 'f_value x_k_plus_1 is: ' + str(f_value)
		print ('f_value x_k_plus_1 is: ' + str(f_value))
		outfile.write(str(log_tmp_2)+'\n')



		#f_value = f_func(A,x_k,b , trig)
		#print ('f_value x_k is: ' + str(f_value))
		
		
		#print ('f_value '+ str(f_value))

		m_value = m_func(A,x_k, b, step_size, x_k_plus_1 , lmd , '2')

		#log_tmp_3 = 'm_value x_k_plus_1 is: '+ str(m_value)
		#print ('m_value x_k_plus_1 is: '+ str(m_value))
		#outfile.write(str(log_tmp_3)+'\n')

		#if f_value <= m_value:
		#	break



		#step_size = step_size * 0.5



		#print ('********************************************')

		#if i%5 == 0:
			#print ('f_value is: ' + str(f_value))
			#print ('m_value is: ' + str(m_value))


		#print('x_k_plus_1 is:' + str(x_k_plus_1))


		if f_func(A,x_k,b, trig) == f_func(A,x_k_plus_1,b , trig):
			break

		elif f_value == 'inf':
			break

		
		x_k = x_k_plus_1


	
	time_2 = time.time()
	print('\n##### No. ' + str(i) + ' ########\n x_k is: \n' + str(x_k))
	print( 'Problem: LASSO\nStep size: ' + str(step_size) + '\nMatrix size: ' + str(n) + ' x ' + str(p) + '\n' + 'Method: ISTA\nRunning time: ' + str(time_2 - time_1) + '\n' )
	#print( 'Final x_k is: ' + str(x_k))
	#print( 'Final x_k_plus_1 is: ' + str(x_k_plus_1) + '\n')
	#print('A2 is ' + str(A))








"""
def main_operation_learning():
	# y ----->  核心计算步骤
	# 通过理论推导出的公式来迭代
	# 比较当前与目标函数的值？
	# opt_x ------> sparse x vector with 20 non-zero elements
	# x_k   ------> preparing for iteration
	# b     ------> sample created by A x opt_x + e
	time_1 = time.time()
	A      = np.random.normal(0,1,(n,p))	#0-1高斯分布
	opt_x  = np.array([0.0]*p)			# optimum x, used for generating b, and try to get optimum x using b and A

	random_index_list = random.sample(range(p), s)
	for i in random_index_list:
		opt_x[i] = np.random.normal(0,10)

	e   = np.random.normal(0,1,n)
	b   = np.dot(A,opt_x.T) + e.T
	x_k = np.array([0.0]*p)
	lmd = math.sqrt(2*n*math.log(p))

	step_length = 10

	for j in range(100):
		obj_value = obj_func(A,x_k,b,lmd)
		#print('obj_value is: ' + str(obj_value))

		#outfile.write(str(obj_value)+'\n')
		for i in range(p):
			while(1):
				#print 'x_k',x_k
				#y = gradient_descent(x_k, step_length, A, b)
				#print 'y',y
				#x_k_plus_1 = prox_operation(lmd,step_length, y)
				#print 'x_k_plus_1',x_k_plus_1
				A_copy = np.copy(A)
				x_copy = np.copy(x_k)
				#A_copy[:,i] = A_copy[:,i] * 0.0
				#x_copy[i] = 0
				#y = x_copy[i] - step_length* ( np.dot( (A[:,i]).T, b-np.dot(A_copy, x_copy.T) ) / np.dot((A[:,i]).T, A[:,i]))


				#print('A[:,i] is: ' + str(A[:,i]))
				#print('A[:,i] length is: ' + str(len(A[:,i])))

				y = np.dot((A[:,i]).T, b - np.dot(A_copy,x_copy.T))/np.dot((A[:,i]).T, A[:,i]) + x_copy[i]


				#new_x_i = prox_operation(lmd, step_length, y)

				x_k_plus_1 = np.copy(x_k)
				x_k_plus_1[i] = y
				f_value = f_func_MSE(A,x_k_plus_1,b)
				#print ('f_value '+ str(f_value))
				m_value = m_func(A,x_k, b, step_length, x_k_plus_1)
				#print ('m_value '+ str(m_value))
				if f_value <= m_value:
					break

				step_length = step_length * 0.5
				#print ('********************************************')
			if f_func_MSE(A,x_k,b) <= f_func_MSE(A,x_k_plus_1,b):
				break


			else:
				x_k = x_k_plus_1

	'''
	print ('m_value '    + str(m_value))
	print ('f_value '    + str(f_value))
	print ('x_k_plus_1 ' + str(x_k_plus_1))
	#print ('x_k[0] ' + str(x_k[0]))
	'''

	time_2 = time.time()
	print( 'Problem: LASSO\nMatrix size: ' + str(n) + ' x ' + str(p) + '\n' + 'Method: Learning\nRunning time: ' + str(time_2 - time_1) + '\n' )
"""



#--------------------------------------------------------------------------------
if __name__ == '__main__':
	#h_sign(1)
	init_1()
	#main_operation_cod()
	main_operation_ista()
	#main_operation_learning()




	#testing:
	'''
	m = np.dot(A.T , A)
	n = np.linalg.eig(m)



	print(m)
	print(n[0])
	print(len(n[0]))


	p = [1,2,3,4,1,5,6,1,6,2,0,5,1,0]
	print(p)
	print(h_sign(p))
	'''
