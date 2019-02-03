from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.io as sio

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


def numOutputPrediction(x):
	for i in range(0,len(x)):
		if(x[i]>0.7):
			return i+1
	return 'No Prediction Made'


def outputTargetImage(y,net_top):
	out = np.zeros(net_top[len(net_top)-1])
	out[y-1]=1
	return out

class Network:

	def __init__(self,fn,nt,a,b,l1):
		self.featureNumber = fn
		self.net_top = nt
		self.alpha = a
		self.bias = b
		self.lambda_1 = l1
		self.theta = self.buildTheta()
		self.a_lj = self.networkConstruct()

	def classificaionContour(self,xmin,xmax,step):
		#contour = np.zeros(np.divide((np.abs(xmin)+xmax),step)*np.divide((np.abs(xmin)+xmax),step))
		contour = np.zeros(int((2/0.1)*(2/0.1)))

		a_lj = self.networkConstruct()
		counter = 0

		for x in np.arange(xmin,xmax,step):
			for y in np.arange(xmin,xmax,step):
				ar = np.array([x,y])
				contour[counter] = self.forwardProp(ar)[len(self.net_top)-1]
				counter+=1

		return np.transpose(np.reshape(contour,(20,20)))



	def buildTheta(self):
		#This is a list not numpy array may cause problems later.
		theta = []
		for i in range(1,len(self.net_top)):
			theta.append(np.random.rand(self.net_top[i],self.net_top[i-1]+1))
		return theta

	def deltaInit(self):
		delta = []  
		for i in range(1,len(self.net_top)):
			delta.append(np.zeros((self.net_top[i],self.net_top[i-1]+1,)))

		return delta

	def sigInit(self):
		sig = []
		for i in range(1,len(self.net_top)):
			sig.append(np.zeros((self.net_top[i])))
		return sig 

	def networkConstruct(self):
		a_lj = []
		#Input
		print(self.net_top[0])
		a_lj.append(np.ones(self.net_top[0]))
		#Hidden
		for i in range(1,len(self.net_top)-1):
			a_lj.append(np.ones(self.net_top[i]))
		#Output
		a_lj.append(np.ones(self.net_top[len(self.net_top)-1]))
		return a_lj
	
	def sigmoid(self,theta,x):
		return 1/(1+np.exp(np.sum((-1*theta*x),axis=1)))

	def tanh(self,theta,x):
		exp = np.sum((theta*x),axis=1)
		return (np.exp(exp)-np.exp(-1*exp))/(np.exp(exp)+np.exp(-1*exp))

	def costFuncLogisticReg(self,y):

		h_theta = self.a_lj[len(self.a_lj)-1]
		logSig = np.log(h_theta)
		OneLogSig = np.log(1-h_theta)
		#log(1-x) tends to -inf at x = 1, python put this as -inf so filter to replace this with large number
		OneLogSig[OneLogSig==np.log(0)] = -10000000000000
		logSig[OneLogSig==np.log(0)] = -100000000000000
		
		#regularizationTerm = (self.lambda_1/(2*self.m))*np.sum(self.theta[1:])
		#s_i = np.sum((-y[img][1]*logSig)-(1-y[img][1])*OneLogSig)
		s_i = np.sum((-y*logSig)-(1-y)*OneLogSig)
		return s_i

	def prediction(self):
		return self.a_lj[len(a_lj)-1]

	def forwardProp(self,x):

		self.a_lj[0] = x 
		#print("theta",self.theta)
		for i in range(1,len(self.theta)+1):
			#Add Bias to all except output
			a_lj_i = self.a_lj[i-1]
			if(i!=len(self.theta)+1):
				a_lj_i = np.insert(a_lj_i, 0, self.bias, axis=0)

			new_a_lj = self.sigmoid(self.theta[i-1],a_lj_i)

			#Updte a_lj
			self.a_lj[i] = new_a_lj

		return self.a_lj
	
	def backProp(self,y,delta):

		sig = self.sigInit()
		dif = self.deltaInit()
		
		# -1 so using 0 works
		netLen = len(self.a_lj)-1

		#Output layer
		sig[len(sig)-1] = -1.0*(y-self.a_lj[netLen])*np.multiply((self.a_lj[len(self.a_lj)-1]),(1.0-self.a_lj[len(self.a_lj)-1]))
		
		#Remove Bias layer in theta 
		theta_NoBias = []
		for i in range(len(self.theta)):
			temp = self.theta[i].copy()
			theta_NoBias.append(np.delete(temp,0,axis=1))

		for l in range(len(sig)-1,0,-1):
			a1 = (np.transpose(theta_NoBias[l]).dot(sig[l]))	
			a2 =  np.multiply(self.a_lj[l],(1.0-self.a_lj[l]))
			a3 = np.multiply(a1 , (a2))
			sig[l-1] = a3

		for i in range(len(delta)-1,-1,-1):
			a_lj_i = np.insert(self.a_lj[i], 0, self.bias, axis=0)
			trans = np.transpose(a_lj_i)
			outer = np.outer(sig[i],trans)
			delta[i] = (delta[i] +outer)

		return delta

	def updateTheta(self):

		for i in range(0,len(self.theta),1):	
			regSum = self.lambda_1*np.sum(self.theta[i][1:len(self.theta[i])-1])
			
			#Reg term acounts for j=0 and j!=0
			theta_temp_j = self.theta[i][:,1:] - self.alpha*(1.0/self.m)*(self.delta[i][:,1:]+np.multiply(self.lambda_1,self.theta[i][:,1:]))			
			theta_temp_jZero = self.theta[i][:,0] - self.alpha*(1.0/self.m)*(self.delta[i][:,0])
			theta_temp = np.insert(theta_temp_j,0,theta_temp_jZero,axis=1)

			self.theta[i] = theta_temp
	
	def trainNN(self,x_train,y_train,cycles):
		self.m = len(x_train)
		for c in range(0,cycles):
			cost=[]
			self.delta = self.deltaInit()
			for m_i in range(0,len(x_train)):
				x = x_train[m_i]
				self.a_lj = self.forwardProp(x)
				self.delta = self.backProp(y_train[m_i],self.delta)
				cost_i = self.costFuncLogisticReg(y_train[m_i])
				cost.append(cost_i)
			print("Epoch",c,"Loss:",(0.5/self.m)*np.sum(cost))

			if (c%10==0):
				contour = NN.classificaionContour(-1,1,0.1)
				plt.imshow(contour, cmap='rainbow', interpolation='nearest')
				plt.pause(0.05)
			self.updateTheta()
			
		plt.show()


from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

dataset = make_circles(n_samples=500,noise=0.1,factor=0.5, random_state=1)

features = dataset[0]
print("Features:",features[:10])
labels = dataset[1]
print("Labels:",labels[:10])

labels = labels.reshape(labels.shape[0],1)

data_len = features.shape[0]

x1_min = np.amin(features[:,0])
x1_max = np.amax(features[:,0])
x2_min = np.amin(features[:,1])
x2_max = np.amax(features[:,1])

X_train = features[:int(data_len*0.7)]
X_test = features[int(data_len*0.3):]
y_train = labels[:int(data_len*0.7)]
y_test = labels[int(data_len*0.3):]


###################################################

###########~~~~~~~~NETWORK SETTINGS & INPUT DATA~~~~~~~~################

###################################################


NN = Network(2,[2,6,1],30,1,0.0)

insert_ = X_train[:,[0]]*X_train[:,[0]]
#X_train = np.append(X_train,insert_,axis=1)

NN.trainNN(X_train,y_train,1000)

contour = NN.classificaionContour(-1.0,1.0,0.1)

plt.imshow(contour, cmap='rainbow', interpolation='nearest')
plt.show()