from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.io as sio

def sigmoid(theta,x):
	return 1/(1+np.exp(np.sum((-1*theta*x),axis=1)))

def costFuncLogisticReg(a_lj,net_top,m,y,theta,lambda_1):

	h_theta = a_lj[len(a_lj)-1]
	logSig = np.log(h_theta)
	OneLogSig = np.log(1-h_theta)
	#log(1-x) tends to -inf at x = 1, python put this as -inf so filter to replace this with large number
	OneLogSig[OneLogSig==np.log(0)] = -10000000000000
	logSig[OneLogSig==np.log(0)] = -100000000000000
	regularizationTerm = (lambda_1/(2*m))*np.sum(theta[1:])
	#s_i = np.sum((-y[img][1]*logSig)-(1-y[img][1])*OneLogSig)
	s_i = np.sum((-y*logSig)-(1-y)*OneLogSig)
	return s_i

def costGradFuncLogisticReg(net_top,m,features,y,theta,lambda_1):
	s = []
	
		#Input
	#x= x_data[img]
		
	#Forward Propergation
	#print(theta,features)
	sig = sigmoid(theta,features)
	logSig = np.log(sig)
	OneLogSig = np.log(1-sig)
	#log(1-x) tends to -inf at x = 1, python put this as -inf so filter to replace this with large number
	OneLogSig[OneLogSig==np.log(0)] = -10000000000000
	logSig[OneLogSig==np.log(0)] = -100000000000000
	#print("cost",logSig.shape,OneLogSig.shape)
	regularizationTerm = (lambda_1/(2*m))*np.sum(theta[1:])
	#s_i = np.sum((-y[img][1]*logSig)-(1-y[img][1])*OneLogSig)
	s_i = np.sum((-y*logSig)-(1-y)*OneLogSig)

	s.append(s_i)
	return (0.5*m)*np.sum(s)+regularizationTerm

def gradientChecking(a_lj,theta,y,lambda_1,m,net_top):
	epsi = 111
	grad_check = []
	a_lj = networkConstruct(net_top)
	for l in range(a_lj[len(a_lj)-1],-1,-1):
		grad = theta[l-1].copy()
		for j in range(0,len(a_lj[l])):
			if(j==len(a_lj[l])-1):
				for i in range(0,len(a_lj[l-1])-1,1):
					for z in range(0,len(theta[l-1][i])-1):
						theta_temp_plus = theta[l-1].copy()
						theta_temp_mins = theta[l-1].copy()
						theta_temp_plus[i][z] = theta_temp_mins[i][z]+epsi
						theta_temp_mins[i][z] = theta_temp_mins[i][z]-epsi
	
						cost = (costGradFuncLogisticReg(net_top,m,a_lj[l-1],y,theta_temp_plus,lambda_1)-costGradFuncLogisticReg(net_top,m,a_lj[l-1],y,theta_temp_mins,lambda_1))/(2*epsi)
						grad[i][z] = cost

			else:
				for i in range(0,len(a_lj[l-1])-1,1):
					for z in range(0,len(theta[l-1][i])-1):

						theta_temp_plus = theta[l-1].copy()
						theta_temp_mins = theta[l-1].copy()
						theta_temp_plus[i] = theta_temp_plus[i]+epsi
						theta_temp_mins[i] = theta_temp_mins[i]-epsi
						cost = (costGradFuncLogisticReg(net_top,m,a_lj[l-1],a_lj[l],theta_temp_plus,lambda_1)-costGradFuncLogisticReg(net_top,m,a_lj[l-1],a_lj[l],theta_temp_mins,lambda_1))/(2*epsi)
						grad[i][z] = cost
		grad_check.append(grad)

	return grad_check 


def classificaionContour(xmin,xmax,step,theta,net_top,bias):
	contour = np.zeros((np.abs(xmin)+xmax)*(np.abs(xmin)+xmax))
	a_lj = networkConstruct(net_top)
	counter = 0
	for x in np.arange(xmin,xmax,1):
		for y in np.arange(xmin,xmax,1):
			ar = np.array([1.0,np.divide(x,100.0),np.divide(y,100.0),np.divide(x*x,100.0),np.divide(y*y,100.0)])
			contour[counter] = forwardProp(a_lj,theta,ar,bias)[len(net_top)-1]
			counter+=1
	return np.transpose(np.reshape(contour,(200,200)))

def buildTheta(net_top,bias):
	#This is a list not numpy array may cause problems later.
	theta = []
	for i in range(1,len(net_top)):
		theta.append(np.random.rand(net_top[i],net_top[i-1]))
		theta[i-1] = np.insert(theta[i-1], 0, bias, axis=1)
	return theta

def numOutputPrediction(x):
	for i in range(0,len(x)):
		if(x[i]>0.7):
			return i+1
	return 'No Prediction Made'

def networkConstruct(net_top):
	a_lj = []
	#Input
	a_lj.append(np.ones(net_top[0]))
	#Hidden
	for i in range(1,len(net_top)-1):
		a_lj.append(np.ones(net_top[i]))
	#Output
	a_lj.append(np.ones(net_top[len(net_top)-1]))
	return a_lj

def outputTargetImage(y,net_top):
	out = np.zeros(net_top[len(net_top)-1])
	out[y-1]=1
	return out

def deltaInit(net_top):
	delta = []
	for i in range(1,len(net_top)):
		delta.append(np.zeros((net_top[i],net_top[i-1]+1,)))
	return delta 

def sigInit(net_top):
	sig = []
	for i in range(1,len(net_top)):
		sig.append(np.zeros((net_top[i])))
	return sig 

def forwardProp(a_lj,theta,x,bias):
	a_lj[0] = x 
	for i in range(1,len(theta)+1):
		new_a_lj = sigmoid(theta[i-1],a_lj[i-1])
		#Add Bias to all except output
		if(i!=len(theta)):
			new_a_lj = np.insert(new_a_lj, 0, bias, axis=0)
		#Updte a_lj
		a_lj[i] = new_a_lj
	return a_lj	

def backProp(a_lj, y, theta,net_top,m,delta):

	sig = sigInit(net_top)
	dif = deltaInit(net_top)
	
	# -1 so using 0 works
	netLen = len(a_lj)-1 
	#Output layer
	sig[len(sig)-1] = -1.0*(y-a_lj[netLen])*np.multiply((a_lj[len(a_lj)-1]),(1.0-a_lj[len(a_lj)-1]))
	#Remove Bias layer in theta and a_lj (a_lj removal skips last entry as output has no bias)
	
	theta_NoBias = []
	for i in range(len(theta)):
		temp = theta[i].copy()
		theta_NoBias.append(np.delete(temp,0,axis=1))

	a_lj_NoBias = []
	for i in range(0,len(a_lj)-1):
		temp = a_lj[i].copy()
		a_lj_NoBias.append(np.delete(temp,0,axis=0))

	for l in range(len(sig)-1,0,-1):
		a1 = (np.transpose(theta_NoBias[l]).dot(sig[l]))		
		a2 =  np.multiply((a_lj_NoBias[l]),(1.0-a_lj_NoBias[l]))
		a3 = np.multiply(a1 , (a2))
		sig[l-1] = a3

	for i in range(len(delta)-1,-1,-1):
		trans = np.transpose(a_lj[i])
		outer = np.outer(sig[i],trans)
		delta[i] = (delta[i] +outer)

	return theta,delta



###################################################

###########~~~~~~~~SETTINGS~~~~~~~~################

###################################################

feat_num = 4
net_top = [feat_num,4,1]
alpha = 1
bias = 1
lambda_1 = 1


###Input Images Data
mat_cont = sio.loadmat('ex3data1.mat')
mat_theta = sio.loadmat('ex3weights.mat')

#Insert adds 1 for x0 and theta0 componsnt (bias)
x_data = np.insert(mat_cont['X'], 0, bias, axis=1)
y_data = np.insert(mat_cont['y'], 0, bias, axis=1)


####Input Circle Data#######
x,n,y_data = np.genfromtxt('ex2data2.txt',delimiter=',',unpack=True)
x_data = np.array((np.ones(len(x)),np.divide(x,100.0),np.divide(n,100.0),np.divide(n*n,100.0),np.divide(x*x,100.0)))
x_data = np.transpose(x_data)

####Calculated Parameters#########
m =len(x_data)

####Theta Construction
theta = buildTheta(net_top,bias)
#theta[0] = mat_theta['Theta1']+0.1
#theta[1] = mat_theta['Theta2']+0.2

#Network Construction
a_lj = networkConstruct(net_top)

for i in range(0,100):
	
	count = 0
	delta = deltaInit(net_top)
	cost = []
	
	for img in range(0,len(x_data)):
		
		x = x_data[img]

		#Forward Propergation
		a_lj = forwardProp(a_lj,theta,x,bias)
		
		#Predictions
		prediction = numOutputPrediction(a_lj[len(a_lj)-1])
		label = y_data[img]

		#BackPropergation
		theta,delta = backProp(a_lj, y_data[img], theta,net_top,m,delta)
		
		#gradCheck = gradientChecking(a_lj,theta,y_data[img],lambda_1,m,net_top)
		#for ch in range(0,len(theta)):
			#print(delta[ch]-gradCheck[ch])

		cost_i = costFuncLogisticReg(a_lj,net_top,m,y_data[img],theta,lambda_1)
		
		if prediction==label:
			count+=1
		
		cost.append(cost_i)

	#Update Theta 
	for i in range(0,len(theta),1):	
		regSum = lambda_1*np.sum(theta[i][1:len(theta[i])-1])
		#Reg term shgould acount for j=0 and j!=0

		theta_temp = theta[i] - alpha*(1.0/m)*(delta[i]+np.multiply(lambda_1,theta[i]))
		theta[i] = theta_temp
	
	
	#Cost Function
	print("Cost:",(0.5/m)*np.sum(cost),count)

contour = classificaionContour(-100,100,1,theta,net_top,bias)
plt.imshow(contour, cmap='rainbow', interpolation='nearest')
plt.show()