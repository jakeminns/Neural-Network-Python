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
		contour = np.zeros((np.abs(xmin)+xmax)*(np.abs(xmin)+xmax))
		a_lj = self.networkConstruct()
		counter = 0
		for x in np.arange(xmin,xmax,1):
			for y in np.arange(xmin,xmax,1):
				ar = np.array([1.0,np.divide(x,100.0),np.divide(y,100.0),np.divide(x*x,100.0),np.divide(y*y,100.0)])
				contour[counter] = self.forwardProp(ar)[len(self.net_top)-1]
				counter+=1
		return np.transpose(np.reshape(contour,(200,200)))



	def buildTheta(self):
		#This is a list not numpy array may cause problems later.
		theta = []
		for i in range(1,len(self.net_top)):
			theta.append(np.random.rand(self.net_top[i],self.net_top[i-1]))
			theta[i-1] = np.insert(theta[i-1], 0, self.bias, axis=1)
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
		a_lj.append(np.ones(self.net_top[0]))
		#Hidden
		for i in range(1,len(self.net_top)-1):
			a_lj.append(np.ones(self.net_top[i]))
		#Output
		a_lj.append(np.ones(self.net_top[len(self.net_top)-1]))
		return a_lj
	
	def sigmoid(self,theta,x):
		return 1/(1+np.exp(np.sum((-1*theta*x),axis=1)))

	def costFuncLogisticReg(self,y):

		h_theta = self.a_lj[len(self.a_lj)-1]
		logSig = np.log(h_theta)
		OneLogSig = np.log(1-h_theta)
		#log(1-x) tends to -inf at x = 1, python put this as -inf so filter to replace this with large number
		OneLogSig[OneLogSig==np.log(0)] = -10000000000000
		logSig[OneLogSig==np.log(0)] = -100000000000000
		regularizationTerm = (self.lambda_1/(2*self.m))*np.sum(self.theta[1:])
		#s_i = np.sum((-y[img][1]*logSig)-(1-y[img][1])*OneLogSig)
		s_i = np.sum((-y*logSig)-(1-y)*OneLogSig)
		return s_i

	def prediction(self):
		return self.a_lj[len(a_lj)-1]

	def forwardProp(self,x):
		self.a_lj[0] = x 
		for i in range(1,len(self.theta)+1):
			new_a_lj = self.sigmoid(self.theta[i-1],self.a_lj[i-1])
			#Add Bias to all except output
			if(i!=len(self.theta)):
				new_a_lj = np.insert(new_a_lj, 0, self.bias, axis=0)
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
		#Remove Bias layer in theta and a_lj (a_lj removal skips last entry as output has no bias)
		
		theta_NoBias = []
		for i in range(len(self.theta)):
			temp = self.theta[i].copy()
			theta_NoBias.append(np.delete(temp,0,axis=1))

		a_lj_NoBias = []
		for i in range(0,len(self.a_lj)-1):
			temp = self.a_lj[i].copy()
			a_lj_NoBias.append(np.delete(temp,0,axis=0))

		for l in range(len(sig)-1,0,-1):
			a1 = (np.transpose(theta_NoBias[l]).dot(sig[l]))		
			a2 =  np.multiply((a_lj_NoBias[l]),(1.0-a_lj_NoBias[l]))
			a3 = np.multiply(a1 , (a2))
			sig[l-1] = a3

		for i in range(len(delta)-1,-1,-1):
			trans = np.transpose(self.a_lj[i])
			outer = np.outer(sig[i],trans)
			delta[i] = (delta[i] +outer)

		return delta

	def updateTheta(self):
		for i in range(0,len(self.theta),1):	
			regSum = self.lambda_1*np.sum(self.theta[i][1:len(self.theta[i])-1])
			#Reg term shgould acount for j=0 and j!=0
			theta_temp = self.theta[i] - self.alpha*(1.0/self.m)*(self.delta[i]+np.multiply(self.lambda_1,self.theta[i]))
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
			print("Cost:",(0.5/self.m)*np.sum(cost))
			self.updateTheta()

###################################################

###########~~~~~~~~NETWORK SETTINGS & INPUT DATA~~~~~~~~################

###################################################

NN = Network(4,[4,4,1],1,1,1)

x1,x2,y_data = np.genfromtxt('ex2data2.txt',delimiter=',',unpack=True)
x_data = np.array((np.ones(len(x1)),np.divide(x1,100.0),np.divide(x2,100.0),np.divide(x2*x2,100.0),np.divide(x1*x1,100.0)))
x_data = np.transpose(x_data)

NN.trainNN(x_data,y_data,100)

contour = NN.classificaionContour(-100,100,1)
plt.imshow(contour, cmap='rainbow', interpolation='nearest')
plt.show()