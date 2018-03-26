# self implemented Neural Network (fully connected NN)
# imports
import numpy as np
import pandas as pd
import cv2
import math

# get data
URL_ENDPOINT = "http://cs.mcgill.ca/~ksinha4/datasets/kaggle/"
train_y = pd.read_csv(URL_ENDPOINT+'train_y.csv', sep=',', header=None, encoding='utf-8')
train_x = pd.read_csv(URL_ENDPOINT+'train_x.csv', sep=',', header=None, encoding='utf-8')

# preprocessing 
def minBoundingCirclePreProcTrain():
  train_x[train_x<240] = 0
  train_x[train_x>=240] = 255

  xImgFiltTrain = train_x.as_matrix().reshape(-1, 64, 64)

  for i,image in enumerate(xImgFiltTrain):
	  image = xImgFiltTrain[i].astype('uint8')

	  ret, thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
	  contourImage, contours, hierarchy = cv2.findContours(thresh,1,2)

	  largest_areas = sorted(contours, key=lambda cont: cv2.minEnclosingCircle(cont)[1])

	  cnt=largest_areas[-1]
	  
	  cnt=largest_areas[-1]

	  (x,y),radius = cv2.minEnclosingCircle(cnt)
	  center = (int(x), int(y))
	  radius = int(math.ceil(radius))
	  mask = np.zeros(image.shape, np.int8)
	  circle = cv2.circle(mask, center, radius, (255,255,255),-1)
	  
	  preprocessed_img = cv2.bitwise_and(image, image, mask=circle)
	  new_dataset_train.append(preprocessed_img)

# activation func
def sigmoid (x):
	#print (np.exp(-x))
	return 1./(1 + np.exp(-x))

# needed for back prop
def derivatives_sigmoid(x):
	return x * (1 - x)

# Leaky relu act
def relu(x):
	# leaky relu?
	return np.maximum(x, 0.000001)

def derive_relu(x):
  return np.greater(x, 0.000001).astype(int)

# helper to one hot encode the training labels
def one_hot(y):
  num_classes = 10
  one_hot_enc = np.zeros((len(y),num_classes))
  
  for i in range(len(y)):
	index = y[0][i]
	np.put(one_hot_enc[i],index,1)
  
#   print (one_hot_enc.shape)
  return one_hot_enc

def calc_error(y_pred,y_true):
  # assume y_pred is shape: len(y) X 10
  y = one_hot(y_true)
  temp = np.subtract(y_pred,y)
  error_vec = np.multiply(temp,temp)/2
  
  return error_vec

# Helper function to predict an output
def predict(probs):
  return np.argmax(probs, axis=1)

if __name__ == "__main__":
	# minBoundingCirclePreProcTest()
	new_dataset_train = []
	minBoundingCirclePreProcTrain()
	new_dataset_train = np.array(new_dataset_train)
	new_dataset_train = new_dataset_train.reshape(-1,4096)
	print (new_dataset_train.shape)
	# cross validation
	# 80% train, 20% validation
	cv_train_x = new_dataset_train[:40000]
	cv_train_y = train_y[:40000]
	cv_valid_x = new_dataset_train[40000:]
	cv_valid_y = train_y[40000:]


	# variable/ parameters inits
	epoch = 100
	lr = 0.1 # param
	num_hidden_layers = 1 # param
	inputlayer_neurons = 4096 #num feats of dataset
	hiddenlayer_neurons = 30 # param
	outputlayer_neurons = 1

	# weights and bias inits
	w1 = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
	b1 = np.random.uniform(size=(1,hiddenlayer_neurons))
	w2 = np.random.uniform(size=(hiddenlayer_neurons,outputlayer_neurons))
	b2 = np.random.uniform(size=(1,outputlayer_neurons))

	# input to be fed into network
	X = np.divide(cv_train_x,float(255))
	y_true = one_hot(cv_train_y)

	# learning rate tuning
lr_list = np.linspace(0.01,0.1,10)
for lr in lr_list:
  print (lr)
  for i in range(epoch):
	# fwd prop
	hidden_layer_input = np.dot(X,w1) + b1
	act_hidden_layer = sigmoid(hidden_layer_input)
	output_layer_input = np.dot(act_hidden_layer,w2) + b2
	y_pred = sigmoid(output_layer_input)

	# back prop
	par_E_y = np.subtract(y_pred,y_true)
	par_y_b = derivatives_sigmoid(y_pred)
	par_b_w2 = act_hidden_layer
	temp_1 = np.multiply(par_E_y,par_y_b)
	del_w2 = np.dot(par_b_w2.T, temp_1)
	del_b2 = np.sum(temp_1,axis=0,keepdims=True)

	temp_2 = np.dot(temp_1,w2.T) #essentially: (par_E_y * par_y_b) dot w2 (the dot with w2 is actually from par_b_a)...
	temp_3 = np.multiply(temp_2,derivatives_sigmoid(act_hidden_layer)) # so far: par_E_y * par_y_b * par_b_a
	del_w1 = np.dot(X.T,temp_3)
	del_b1 = np.sum(temp_3, axis=0,keepdims=True)

	# update weights (Grad Desc)
	w1 += del_w1 * lr
	b1 += del_b1 * lr
	w2 += del_w2 * lr
	b2 += del_b2 * lr
#     print (i)
  hidden_layer_input = np.dot(cv_valid_x,w1) + b1
  act_hidden_layer = sigmoid(hidden_layer_input)
  output_layer_input = np.dot(act_hidden_layer,w2) + b2
  output = sigmoid(output_layer_input)
  p = predict(output)
  correct = 0
  for i in range(len(p)):
	if (p[i] == cv_valid_y[0][40000+i]):
	  correct +=1
  accuracy = float(correct/len(cv_valid_y))
  print ("accuracy: ", accuracy)
