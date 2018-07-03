import numpy as np
from copy import deepcopy

class TransE:
	
	def __init__(self, entityList, relationList, tripleList, margin = 1, learningRate = 0.00001, dim = 10, L1 = True):
		self.margin=margin
		self.learningRate=learningRate
class Foo():
	x=9
	def __init__(self,x=1):
		self.x=x
	def fu(self,b):
		self.x=b



def init(dim):
	'''
	初始化向量
	'''
	return np.random.uniform(-6/(dim**0.5),6/(dim**0.5))

def distanceL1(h,l,t):
	s=h+l-t
	sum=np.fabs(s).sum()
	return sum

def distanceL2(h,l,t):
	s=h+l-t
	sum=(s*s).sum()
	return sum

def norm(list):
	'''
	归一化
	'''
	var=np.linalg.norm(list)
	#norm([3,4])=5
	i=0
	while i<len(list):
		list[i]=list[i]/var
		i=i+1
	return np.array(list)

def openDetailsAndId(dir,sp=','):
	idNum=0
	list=[]
	with open(dir) as file:
		lines=file.readlines()
		for line in lines:
			DetailsAndId=line.strip().split(sp)
			list.append(DetailsAndId[0])
			idNum+=1
	return idNum,list

def openTrain(dir,sp=','):
	num=0
	list=[]
	with open(dir) as file:
		lines=file.readlines()
		for line in lines:
			triple=line.strip().split(sp)
			if(len(triple)<3):
				continue
			list.append(tuple(triple))
			num+=1
	return num,list

if __name__ == '__main__':
	dirEntity="data\\people.csv"
	a,b=openDetailsAndId(dirEntity)
	print(a,b)