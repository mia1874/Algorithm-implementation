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
						2. 
						3. 
		''')


#--------------------------------variables----------------------------------------------
# size parameters
'''
n = 1000		# row size
p = 500		# column size
'''
n = 500		# row size
p = 250		# column size
s = 20		# nonzero size
step_length = 10
lmd = math.sqrt(2*n*math.log(p))  #lambda in LOSSA objective function







#----------------------------------backup----------------------------------------------
# preparing for using 
'''
def sign(y):
	if y > 0.0:
		return 1.0
	elif y < 0.0:
		return -1.0
	else:
		return 0.0

#compare a value with each element in a vector
# and replace the element with the max one, which is the value of origin 
#return a new vector
def vec_max(value, vector):
	for i in range(len(vector)):
		vector[i] = max(value, vector[i])
	return vector


#proximal function 
#lmd is the lambda value for convex closed function
# y is the [x_k - r_k* f'(x_k)]
#output prox_vec is p*1 dimensions
def prox_operation(lmd,step_length,y):
	new_y = sign(y) 
	print('new_y ' + str(new_y))
	prox_value = new_y * max(0, abs(y) - step_length*lmd)   # NOTE: here is step_length * lmd, use max instead vec_max function 
	print('prox '+ str(prox_value))
	return prox_value
# gradient step
def gradient_descent(x_k, step_length,A,b):
	f_prime = np.dot(A.T, np.dot(A,x_k.T) - b) # derivertive of f_function 
	print('f_prime '+ str(f_prime))
	y = x_k - step_length * f_prime
	return y 

'''



#---------------------------------initilization------------------------------------------    
'''
def init():
	# when initialize a vector, had better to initialize it as row vector instead column vector  
	global A
	global opt_x
	global e
	global b
	global x_k
	global lmd
	global random_index_list

	# opt_x ------> sparse x vector with 20 non-zero elements
	# x_k   ------> preparing for iteration
	# b     ------> sample created by A x opt_x + e
	A     = np.random.normal(0,1,(n,p))		#0-1高斯分布
	opt_x = np.array([0.0]*p)			# optimum x, used for generating b, and try to get optimum x using b and A

	random_index_list = random.sample(range(p), s) 
	for i in random_index_list:
		opt_x[i] = np.random.normal(0,10)
	
	e   = np.random.normal(0,1,n)      
	b   = np.dot(A,opt_x.T) + e.T
	x_k = np.array([0.0]*p)
	lmd = math.sqrt(2*n*math.log(p))

	print ('opt_x '+ str(opt_x))
	print ('e '+ str(e))
	print('opt_x is:' + str(opt_x))
	print('length opt_x is:' + str(len(opt_x)))
	print ('b '+ str(b))
	print ('lambda'+ str(lmd))

'''


#----------------------------------main function-----------------------------------------
# primary function f(x) = (||Ax - b||^2) /2 
def f_func(A,x,b):
	f_value = math.pow(np.linalg.norm(np.dot(A,x.T) - b),2) / 2
	return f_value

#model function
def m_func(A, x_n, b, s, x):
	part_1 = f_func(A,x_n,b)
	#part_1 = math.pow(np.linalg..norm(np.dot(A,x_k.T) - b),2) / 2 
	#print ('part_1 '+ str(part_1))
	part_2 = np.dot( (np.dot(A.T, np.dot(A,x_n.T) - b)).T, (x-x_n).T )
	#print ('part_2 '+ str(part_2))
	part_3 = math.pow(np.linalg.norm(x-x_n),2) / (2*s)
	#print ('part_3 '+ str(part_3))
	m_value = part_1 + part_2 + part_3
	#print('m_value '+ str(m_value))
	return m_value






#objective function
# 这个lmb和1范数根本没用上，优化的就是一个LASSO 
# 这个obj_func未使用
# 优化的还是f_func 与 m_func 之间
def obj_func(A,x,b,lmd):
	#f_value = math.pow(np.linalg.norm(np.dot(A,x.T) - b),2) / 2
	f_value = f_func(A,x,b)
	g_value = lmd * np.linalg.norm(x,1)
	#value = f_value + g_value
	value = f_value

	return value








#----------------------------------main operation-----------------------------------------

def main_operation_cod():
	# y ----->  核心计算步骤
	# 通过理论推导出的公式来迭代
	# 比较当前与目标函数的值？
	# opt_x ------> sparse x vector with 20 non-zero elements
	# x_k   ------> preparing for iteration
	# b     ------> sample created by A x opt_x + e
	time_1 = time.time()
	A     = np.random.normal(0,1,(n,p))	#0-1高斯分布
	opt_x = np.array([0.0]*p)			# optimum x, used for generating b, and try to get optimum x using b and A

	random_index_list = random.sample(range(p), s) 
	for i in random_index_list:
		opt_x[i] = np.random.normal(0,10)

	e   = np.random.normal(0,1,n)      
	b   = np.dot(A,opt_x.T) + e.T
	x_k = np.array([0.0]*p)
	lmd = math.sqrt(2*n*math.log(p))

	step_length = 10


	'''
	print ('opt_x '+ str(opt_x))
	print ('e '+ str(e))
	print('opt_x is:' + str(opt_x))
	print('length opt_x is:' + str(len(opt_x)))
	print ('b '+ str(b))
	print ('lambda'+ str(lmd))
	'''
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
				f_value = f_func(A,x_k_plus_1,b) 
				m_value = m_func(A,x_k, b, step_length, x_k_plus_1)

				
				
				
				if f_value <= m_value:
					break

				step_length = step_length * 0.5
				#print ('********************************************')
			
			
			#---------------------------debug start----------------------------
			#break条件：1合理阈值 2固定迭代次数
			if f_func(A,x_k,b) <= f_func(A,x_k_plus_1,b)+1e-8:
			#if f_func(A,x_k,b) <= f_func(A,x_k_plus_1,b):
				break
		
			if i%100 == 0:
				print ('f_value is:'+ str(f_value))
				print ('m_value is:'+ str(m_value))


			else:
				x_k = x_k_plus_1
	
	'''
	print ('m_value '    + str(m_value))
	print ('f_value '    + str(f_value))
	print ('x_k_plus_1 ' + str(x_k_plus_1))
	#print ('x_k[0] ' + str(x_k[0]))
	'''

	time_2 = time.time()
	print( 'Problem: LASSO\nMatrix size: ' + str(n) + ' x ' + str(p) + '\n' + 'Method: Coordinate descent\nRunning time: ' + str(time_2 - time_1) + '\n' )


def main_operation_ista():
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
				f_value = f_func(A,x_k_plus_1,b) 
				#print ('f_value '+ str(f_value))
				m_value = m_func(A,x_k, b, step_length, x_k_plus_1)
				#print ('m_value '+ str(m_value))
				if f_value <= m_value:
					break

				step_length = step_length * 0.5
				#print ('********************************************')
			if f_func(A,x_k,b) <= f_func(A,x_k_plus_1,b):
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
	print( 'Problem: LASSO\nMatrix size: ' + str(n) + ' x ' + str(p) + '\n' + 'Method: ISTA\nRunning time: ' + str(time_2 - time_1) + '\n' )




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
				f_value = f_func(A,x_k_plus_1,b) 
				#print ('f_value '+ str(f_value))
				m_value = m_func(A,x_k, b, step_length, x_k_plus_1)
				#print ('m_value '+ str(m_value))
				if f_value <= m_value:
					break

				step_length = step_length * 0.5
				#print ('********************************************')
			if f_func(A,x_k,b) <= f_func(A,x_k_plus_1,b):
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




#--------------------------------------------------------------------------------
if __name__ == '__main__':
	main_operation_cod()
	main_operation_ista()
	#main_operation_learning()



