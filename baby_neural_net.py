import numpy as np 

weights = [.9, .3, .4]

def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred

wl_rec = ([0.3, 0.3, 0.6])
toes = ([9, 8, 7, 4])
hurt = ([.1, .2, 0, 1])

input = wl_rec[0]
true = [toes[0], hurt[0]]

pred = neural_network(input, weights)

error = [0,0,0]
delta = [0,0,0]

for i in range(len(true)):
	error[i] = (pred[i] - true[i] **2)
	delta[i] = (pred[i] - true[i])
#new block
weights = [.8, .3, .4]

def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred

wl_rec = ([0.3, 0.5, 0.6])
toes = ([10, 9, 19])
hurt = ([.1, .2, .3])

input = wl_rec([0.3, 0.3, 0.6])
true = [toes[0], hurt[0]]

pred = neural_network(input, weights)
error = [0,0,0]
delta = [0,0,0]

for i in range(len(true)):
	error[i] = (pred[i] - true[i] **2)
	delta[i] = (pred[i] - true[i])


weights = np.random()
def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred

wl_rec = ([0.9,0.9,0.9])
toes = ([1.0, 1.0, 1.0])
hurt = ([0.1, 0.1, 0.1])

input = wl_rec(0.3, 0.3, 0.3)
true = [toes[0], hurt[0]]

pred = neural_network(input, weights)
error = [0,0,0]
delta = [0,0,0]

for i in range(len(true)):
	error[i] = (pred[i] - true[i] **2)
	delta[i] = (pred[i] - true[i])

#NEW BLOCK NEW BLOCK NEW BLOCK 

weights = np.random() #probably needs length automatically specified

def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred

wl_rec = ([0.1, 0.2, 0.1])
toes = ([0.2, 0.3, 0.3])
hurt = ([10, 20, 30])

input = wl_rec(0.1, 0.2, 0.2)
true = [toes[0], hurt[0]]

pred = neural_network(input, weights)
error = [0,0,0]
delta = [0,0,0]

for i in range(len(true)):
	error[i] = (pred[i] - true[i] **2)
	delta[i] = (pred[i] - true[i])

#NEW NEW NEW NEW NEW 

def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred

wl_rec = ([0.1, 0.3, 0.5])
toes = ([0.3, 0.4, 0.3])
hurt = ([.4, .5, .6])


input = wl_rec(0.2, 0.4, 0.9)
true = [toes[0], hurt[0]]

pred = neural_network(input, weights)
error = [0,0,0]
delta = [0,0,0]

for i in range(len(true)):
	error[i] = (pred[i] - true[i] **2)
	delta[i] = (pred[i] - true[i])






















#new

weight = 0.5
goal_pred = 0.8
input = 0.1
alpha = 0.1

for iteration in range(20):
	pred = input * weight
	error = (pred - goal_pred) **2
	derivative = input * (pred - goal_pred)
	weight = weight - (alpha*derivative)
print("Error:" + str(error)		+ "Prediction:" + str(pred))






weight = 0.5
goal_pred = 0.4
input = 2
alpha = .01

for iteration in range(20):
	pred = input*weight
	error = (pred-goal_pred) **2
	derivative = input * (goal_pred)
	weight = weight - (alpha*derivative)
	print("Error:" + str(error)	+ "Pred:" + str(pred))


weight = 0.5
goal_pred = 0.4
input = 2
alpha = .01

for iteration in range(20):
	pred = input*weight
	error = (pred-goal_pred **2) #this is just mean squared error
	derivative = input * (goal_pred)
	weight = weight - (alpha*derivative)
	print("error:" + str(error) + "Prediction:" + str(pred))


weight = 0.6
goal_pred = 0.6
input = 1
alpha = 0.1

for iteration in range(20):
	pred = input*weight 
	error = (pred - goal_pred **2)
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)
	print("error:" + str(error) + "Prediction:" + str(pred))


weight = 0.1
goal_pred = 0.8
input = 1
alpha = 0.1

for iteration in range(20):
	pred = input*weight
	error = (pred - goal_pred **2)
	derivative = input*goal_pred
	weight = weight - (alpha*derivative)
	print("error:" + str(error) + "prediction:" + str(pred))


weight = 0.2
goal_pred = 0.4 
input = .8
alpha = .4

for iteration in range(20):
	pred = input * weight
	error = (pred - goal_pred **2)
	derivative = input*goal_pred
	weight = weight - (alpha*derivative) #this is the weight UPDATE
	print("error:" + str(error) + "prediction" + str(pred))


weight = 0.2
goal_pred = 0.4
input = 0.4
alpha = 0.0001

for iteration in range(20):
	pred = input*weight
	error = (pred - goal_pred **2)
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)
	print("error:" + str(error) + "prediction:" + str(pred))


weight = 0.2
goal_pred = 0.3
input = 0.4
alpha = 0.0000001

for iteration in range(20):
	pred = input*weight
	error = (pred - goal_pred **2)
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)
	print("error:" + str(error)	+ "prediction:" + str(pred))

weight = 0.3
goal_pred = 0.4
input = 0.4
alpha = 0.01

for iteration in range(30):
	pred = input*weight
	error = (pred - goal_pred **2)
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)
	print("error:"+str(error) + "prediction:"+str(pred))


weight = 0.3
goal_pred = 0.1
input = 0.5
alpha = 0.01

for iteration in range(20):
	pred = input*weight
	error = (pred - goal_pred**2)
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)
	print("Error:" + str(error) + "Prediction:" + str(pred))


for iteration in range(20):
	pred = input*weight
	error = (pred - goal_pred**2)
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)
	print("Error:" + str(error)	+	"Prediction" +str(pred))


for iteration in range(20):
	pred = input*weight
	error = (pred - goal_pred **2)
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)
	print("Error:" + str(error) + "prediction:" +str(pred))

for iteration in range(20):
	pred = input*weight
	error = (pred - goal_pred)**2
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)
	print("error:" + str(error) + "Prediction" + str(pred))



def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred

weights = ([0,1,1,.8])

n_words = ([9, 19, .9, 1]) 
n_sentences = ([1, 10, .9, .1])
n_atoms = ([.2, .1, .4, .6])

input = n_words([9,19,.9,.1])
true = [n_sentences[0], n_atoms[0]]
goal_pred = true[0,0]

for iteration in range(20):
	pred = neural_network(input, weights)
	error = (pred - goal_pred)**2
	derivative = input*(goal_pred)
	weight = weight - (alpha*derivative)

def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
	for iteration in range(20)
		goal_pred = 0.9
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = input*(goal_pred)
		weight = weight - (alpha*derivative)
		print("error:" + str(error) + "Prediction:" + str(pred))





def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
weights = ([0,1,1,.8])

n_words = ([9, 19, .9, 1]) 
n_sentences = ([1, 10, .9, .1])
n_atoms = ([.2, .1, .4, .6])
	for iteration in range(20):
		goal_pred = 0.9
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = input*(goal_pred)
		weight = weight - (derivative*alpha)
		print("error:" + str(error) + "prediction:" + str(pred))












def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred

weights = [0.1, 0.3, 0.4]
toes = [0.1, 0.4, 3, 0.2]
wl_rec = [0.2, 0.1, 0.4, 0.9]
hurt = [0.9, 0.2, 0.7, 1]

input = wl_rec[0] 
true = [toes[0], hurt[0]]
pred = neural_network(input, weights)

for iteration in range(20):
	alpha = 0.01
	goal_pred = 0.9
	error = (pred - goal_pred **2)
	derivative = (input*goal_pred)
	weight = weight - (alpha*derivative)
	print("prediction:" + str(pred) + "error:" + str(error))


def neural_network(input, weights):
	pred = ele_mul(input, weights)

weights = [0.3, 0.3, 0.4]

wl_rec = [0.3, 0.5, 0.6]
toes = [10, 10, 12]
hurt = [.9, .1, 0]

input = wl_rec[0]
true = [toes[0], hurt[0]]
pred = neural_network(input, weights)

for iteration in range(30):
	alpha = 0.02
	goal_pred = 0.9
	error = (pred - goal_pred)**2
	derivative = (input*goal_pred)
	weight = weight - (derivative*alpha)
	print("prediction:" + str(pred) + "error:" + str(error))
















def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
 
	weights = [0.9, 0.8, 0.7]

	wl_rec = [0.8, 0.8, 0.5]
	hurt = [1, 0.1, .2]
	toes = [10, 10, 12]

	input = wl_rec(0.9, 0.8, 0.7) #<- this is a tuple because input values are held constant
	true = [hurt[0], toes[0]]
	pred = neural_network(input, weights)
	goal_pred = 0.9
	for iteration in range(30):
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = input*goal_pred
		weight = weight - (derivative*alpha)
		print("prediction:" + str(pred) + "error:" + str(error))


def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred

	weights = [0.1,0.4, 0.5]

	wl_rec = [0.8, 0.8, 0.5]
	hurt = [1, 0.9, 0.8]
	toes = [.1, .4, .8]

	input = wl_rec(0.9, 0.8, 0.7)
	true = [hurt[0], toes[0]]
	pred = neural_network(input, weights)
	goal_pred = 0.9
	for iteration in range(20):
		alpha = 0.1
		error = (pred - goal_pred)**2
		derivative = (input*goal_pred)
		weight = weight - (derivative*alpha)
		print("prediction:" + str(pred) + "error:" + str(error))
















































def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
	goal_pred = 0.9
	weights = [0.3, 0.4, 0.5]

	wl_rec = [0.3, 0.5, 0.8]
	toes = [1, 2, 10]
	hurt = [1, 0.1, 0.1]

	input = wl_rec(0.3, 0.50, 0.8)
	true = [toes[0], hurt[0]]

	for iteration in range(30):
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = (input*goal_pred)
		weights = weights - (alpha*derivative)
		print("prediction:" + str(pred) + "error:" + str(error))


def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
	goal_pred = 0.8
	weights = [0.9, 0.3, 0.5]

	wl_rec = [0.3, 0.5, 0.6]
	toes = [1, 2, 3]
	hurt = [1.1, 0.9, .1]
	home_game = [0.1, .9, 0.3]

	input = wl_rec[0.3, 0.5, 0.6]
	true = [toes[0], hurt[0], home_game[0]]

	for iteration in range(50):
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = (input*goal_pred)
		weights = weights - (alpha*derivative)
		print("prediction:" + str(pred) + "error:" + str(error))






def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
	goal_pred = 0.9
	weights = [0.9, 0.3, 0.6]

	n_atoms = [4, 3, 2]
	n_sentences = [8, 9, 4]
	n_words = [40, 20, 30]
	truth_g = [0.9, 0.4, 0.75]
	input = n_atoms[4,3,2]
	true = [n_atoms[0], n_sentences[0], n_words[0], truth_g[0]]

	for iteration in range(500):
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = (input*goal_pred)
		weights = weights - (alpha*derivative)
		print("prediction:" + str(pred) + "error:" + str(error))




def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
	goal_pred = 0.75
	weights = [0.9, 1, 0.6]

	n_atoms = [4, 3, 6]
	n_sentences = [8,9,10]
	n_words = [40, 20, 30]
	truth_g = [0.9, 0.5, 0.6]
	input = n_atoms[4,3,2]
	true = [n_atoms[0], n_sentences[0], n_words[0], truth_g[0]]

	for iteration in range(400):
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = (input*goal_pred)
		weights = weights - (alpha*derivative)
		print("prediction:" + str(pred) + "error:" + str(error))




def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
	goal_pred = 0.756
	weights = [0.9, 1, 0.6]

	n_atoms = [4, 3, 6]
	n_sentences = [8, 9, 10]
	n_words = [41, 35, 67]
	truth_g = [0.9, 0.5, 0.7]
	input = n_atoms[4,3,2]
	true = [n_atoms[0], n_sentences[0], n_words[0], truth_g[0]]

	for iteration in range(400):
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = (input*goal_pred)
		weights = weights - (alpha*derivative)
		print("prediction:" + str(pred) + "error" + str(error))


def neural_network(input, weights):
	pred = ele_mul(input, weights)
	goal_pred = 0.89
	weights = [0.9, 0.1, 0.6]

	n_words = [34, 37, 78]
	n_sentences = [8, 9, 10]
	n_atoms = [3, 2, 5]
	truth_g = [0.9, 0.2, 0.1]
	input = n_atoms[4,3,2]
	true = [n_atoms[0], n_sentences[0], n_words[0], truth_g[0]]

	for iteration in range(400):
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = (input*goal_pred)
		weights = weights - (alpha*derivative)
		print("prediction:" + str(pred) + "error:" + str(error))








def neural_network(input, weights):
	pred = ele_mul(input, weights)
	return pred
	goal_pred = 0.8
	weights = 0.7

	wl_rec = [0.3, 0.4, 0.6]
	toes = [10, 9, 12]
	hurt = [1.1, 1.4, 1.8]
	home_game = [0.3, 0.2, 0.4]

	input = wl_rec[0.3, 0.4, 0.5]
	true = [toes[0], hurt[0], home_game[0]]


	for iteration in range(45):
		alpha = 0.01
		error = (pred - goal_pred)**2
		derivative = (input*goal_pred)
		weights = weights - (alpha*derivative)
		print("prediction:" + str(pred) + "error:" + str(error))




import numpy as np
weights = np.array([0.5, 0.8, -0.2])
alpha = 0.01

streetlights = ([[0, 1, 1],
				[0, 0, 1],
				[1, 1, 1],
				[1, 0, 0],
				[1, 1, 1]
				[0, 1, 1]])
walk_vs_stop = np.array([0, 1, 1, 0, 1, 0])

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

for iteration in range(20):
	prediction = input.dot(weights)
	error = (goal_prediction - prediction)**2
	delta = prediction - goal_prediction
	weights = weights - (alpha*(input*delta))
	print("error:" + str(error) + "prediction:" + str(prediction))




import numpy as np
weights = np.array([0.8, 0.5, -0.9])
alpha = 0.01

streetlights = ([[0, 1, 1], 
				 [0, 1, 0],
				 [0, 1, 0],
				 [0, 0, 1]
				 [ 0, 1, 1]
				 [1, 1, 1]])

walk_vs_stop = np.array([0, 1, 1, 0, 1, 0])

input = streetlights[0]
goal_prediction = walk_vs_stop[0]

for iteration in range(20):
	prediction = input.dot(weights)
	error = (goal_prediction - prediction)**2
	delta = prediction - goal_prediction
	weights = weights - (alpha*(input*(delta)))
	print("error:" + str(error) + "prediction:" + str(prediction))



	
































