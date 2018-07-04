import numpy as np
from copy import deepcopy

class TransE:
	def __init__(self, entityList, relationList, tripleList, margin = 1, learningRate = 0.00001, dim = 10, L1 = True):
		self.margin=margin
		self.learningRate=learningRate
		self.dim=dim
		self.L1=L1
		self.entityList=entityList#初始化后，变为字典，key是entity，values是其向量（使用narray）。
		self.relationList=relationList
		self.tripleList=tripleList
		self.loss=0

	def initialize(self):
		entityVectorList={}
		relationVectorList={}
		for entity in entityList:
			n=0
			entityVector=[]
			while n<self.dim:
				ram=init(self.dim)
				entityVector.append(ram)
				n+=1
			entityVector=norm(entityVector)
			entityVectorList[entity]=entityVector
		print('实体向量初始化完成，共有%d个'%len(entityVectorList))
		for relation in relationList:
			n=0
			relationVector=[]
			while n<self.dim:
				ram=init(self.dim)
				relationVector.append(ram)
				n+=1
			relationVector=norm(relationVector)
			relationVectorList[relation]=relationVector
		print('关系向量初始化完成，共有%d个'%len(relationVectorList))
		self.entityList=entityVectorList
		self.relationList=relationVectorList


def init(dim):
	'''
	初始化向量
	'''
	return np.random.uniform(-6/(dim**0.5),6/(dim**0.5))

def distanceL1(h,l,t):
	#L1范数
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
#返回行数和指定
def openDetailsAndId(dir,sp=',',col=0):
	idNum=0
	list=[]
	with open(dir) as file:
		lines=file.readlines()
		for line in lines:
			DetailsAndId=line.strip().split(sp)
			list.append(DetailsAndId[col])
			idNum+=1
	return idNum,list
#返回训练样本数和列表
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
	entityIdNum,entityList=openDetailsAndId(dirEntity,',',1)
	print(entityIdNum,entityList)
	dirRelation = "data\\rel.csv"
	relationIdNum, relationList = openDetailsAndId(dirRelation,',',1)
	print(relationIdNum,relationList)
	#dirTrain = data\\train.txt"
	#tripleNum, tripleList = openTrain(dirTrain)
	print("打开TransE")
	#transE = TransE(entityList,relationList,tripleList, margin=1, dim = 100)
	#print("TranE初始化")
	#transE.initialize()
	#transE.transE(15000)
	#transE.writeRelationVector("c:\\relationVector.txt")
	#transE.writeEntilyVector("c:\\entityVector.txt")
