import numpy as np
import random#random.sample
import math
from copy import deepcopy
import csv
dType='float32'
'''
dim=10
entity_Dict={}
relation_Dict={}
csv_file=csv.reader(open('../data/WN182/entity2id.txt'))
entity_num=0
for lines in csv_file:
	line=lines[0].split('\t')
	entity_num+=1
	entity_Dict[line[0]]=int(line[1])

csv_file=csv.reader(open('../data/WN182/relation2id.txt'))
relation_num=0
for lines in csv_file:
	line=lines[0].split('\t')
	relation_num+=1
	relation_Dict[line[0]]=int(line[1])
print("entity number:%d,relation number:%d"%(entity_num,relation_num))

#tf.placeholder()

e=[]
e_p=[]
for i in range(entity_num):
	e.append(tf.Variable(tf.random_uniform([dim,1],-1.0,1.0)))
	e_p.append(tf.Variable(tf.random_uniform([dim,1],-1.0,1.0)))
	if i%1000==0:
		print('initializing entity:%d/%d'%(i,entity_num))

r=[]
r_p=[]
for i in range(relation_num):
	r.append(tf.Variable(tf.random_uniform([dim,1],-1.0,1.0)))
	r_p.append(tf.Variable(tf.random_uniform([dim,1],-1.0,1.0)))
	print('initializing relation:%d/%d'%(i,relation_num))
'''

class TransD:
	def __init__(self, entityList, relationList, tripletList, margin = 0.001, learningRate = 0.00001, dimE = 10,dimR =10):
		self.margin=margin
		self.learningRate=learningRate
		self.dimE=dimE
		self.dimR=dimR
		self.entityList=entityList#初始化后，变为字典，key是entity，values是其向量（使用narray）。
		self.relationList=relationList
		self.tripletList=tripletList
		self.loss=0
		random.shuffle(self.tripletList)

	def initialize(self):
		entityVectorList={}
		entityMappingList={}
		relationVectorList={}
		relationMappingList={}

		for entity in self.entityList.values():
			n=0
			entityVector=[]
			entityMVector=[]
			while n<self.dimE:
				ram=init(self.dimE)
				entityVector.append(ram)
				ram=init(self.dimE)
				entityMVector.append(ram)
				n+=1
			#归一化 并且类型为np.array
			entityVector=norm(entityVector)
			entityVectorList[entity]=entityVector
			entityMVector=norm(entityMVector)
			entityMappingList[entity]=entityMVector

		print('实体向量初始化完成，共有%d个'%len(entityVectorList))
		for relation in self.relationList.values():
			n=0
			relationVector=[]
			relationMVector=[]
			while n<self.dimR:
				ram=init(self.dimR)
				relationVector.append(ram)
				ram=init(self.dimR)
				relationMVector.append(ram)
				n+=1
			relationVector=norm(relationVector)
			relationVectorList[relation]=relationVector
			relationMVector=norm(relationMVector)
			relationMappingList[relation]=relationMVector

		print('关系向量初始化完成，共有%d个'%len(relationVectorList))
		self.entityList=entityVectorList
		self.entityMappingList=entityMappingList
		self.relationList=relationVectorList
		self.relationMappingList=relationMappingList


	def transD(self,cI=20,batchNum=150):
		print("start training")
		n=math.ceil(len(self.tripletList)/batchNum)
		cI=cI*n

		copyTripletList=deepcopy(self.tripletList)
		for cycleIndex in range(1,cI+1):
			Sbatch=copyTripletList[:batchNum]
			del copyTripletList[:batchNum]
			#Sbatch=self.getSample(batchNum)#######
			Tbatch=[]#triplet and corrupted triplet
			for sbatch in Sbatch:
				tripletAndCorruptedTriplet=(sbatch,self.getCorrupted(sbatch))
				if(tripletAndCorruptedTriplet not in Tbatch):
					Tbatch.append(tripletAndCorruptedTriplet)
			self.update(Tbatch)
			if cycleIndex%n==0:

				copyTripletList=deepcopy(self.tripletList)
				random.shuffle(copyTripletList)
			if cycleIndex%100==0:
				print("the %d th training"%cycleIndex)
				print('loss=',self.loss)
				'''
				self.writeRelationVector()
				self.writeEntilyVector()
				'''
				self.loss=0

	def getSample(self,size):
		return random.sample(self.tripletList,size)

	def getCorrupted(self,triplet):
		i=np.random.uniform(-1,1)
		if i<0:
			while True:
				entityTemp=random.sample(self.entityList.keys(),1)[0]
				corruptedTriplet=(entityTemp,triplet[1],triplet[2])
				if corruptedTriplet not in tripletList:
					break

		else:
			while True:
				entityTemp=random.sample(self.entityList.keys(),1)[0]
				corruptedTriplet=(triplet[0],entityTemp,triplet[2])
				if corruptedTriplet not in tripletList:
					break
			
		return corruptedTriplet

	def update(self,Tbatch):
		copyEntityList=deepcopy(self.entityList)
		copyEntityMappingList=deepcopy(self.entityMappingList)
		copyRelationList=deepcopy(self.relationList)
		copyRelationMappingList=deepcopy(self.relationMappingList)
		for triplets in Tbatch:#triplets[0][0]...[1][2]
			headVector=copyEntityList[triplets[0][0]]
			headMappingVector=copyEntityMappingList[triplets[0][0]]
			tailVector=copyEntityList[triplets[0][1]]
			tailMappingVector=copyEntityMappingList[triplets[0][1]]
			relationVector=copyRelationList[triplets[0][2]]
			relationMappingVector=copyRelationMappingList[triplets[0][2]]

			headVectorCorrupted=copyEntityList[triplets[1][0]]
			headMappingVectorCorrupted=copyEntityMappingList[triplets[1][0]]
			tailVectorCorrupted=copyEntityList[triplets[1][1]]
			tailMappingVectorCorrupted=copyEntityMappingList[triplets[1][1]]

			headVectorBefore=self.entityList[triplets[0][0]]
			headMappingVectorBefore=self.entityMappingList[triplets[0][0]]
			tailVectorBefore=self.entityList[triplets[0][1]]
			tailMappingVectorBefore=self.entityMappingList[triplets[0][1]]
			relationVectorBefore=self.relationList[triplets[0][2]]
			relationMappingVectorBefore=self.relationMappingList[triplets[0][2]]

			headVectorCorruptedBefore=self.entityList[triplets[1][0]]
			headMappingVectorCorruptedBefore=self.entityMappingList[triplets[1][0]]
			tailVectorCorruptedBefore=self.entityList[triplets[1][1]]
			tailMappingVectorCorruptedBefore=self.entityMappingList[triplets[1][1]]

			distTriplet=distanceTransD(headVectorBefore,headMappingVectorBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorBefore,tailMappingVectorBefore)
			distCorruptedTriplet=distanceTransD(headVectorCorruptedBefore,headMappingVectorCorruptedBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorCorruptedBefore,tailMappingVectorCorruptedBefore)
			eg=self.margin+distTriplet-distCorruptedTriplet
			if eg>0:
				self.loss+=eg
				tempPosH=[]
				for i in range(self.dimE):
					temp=2*(distanceTransDL1(headVectorBefore,headMappingVectorBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorBefore,tailMappingVectorBefore)*(relationMappingVectorBefore*headMappingVectorBefore[i]+1)).sum()
					tempPosH.append(temp)
				tempPosH=np.array(tempPosH,dtype=dType).reshape(-1,1)#test

				tempPosHP=[]
				for i in range(self.dimE):
					temp=2*(distanceTransDL1(headVectorBefore,headMappingVectorBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorBefore,tailMappingVectorBefore)*(relationMappingVectorBefore*headVectorBefore[i])).sum()
					tempPosHP.append(temp)
				tempPosHP=np.array(tempPosHP,dtype=dType).reshape(-1,1)

				tempR=2*distanceTransDL1(headVectorBefore,headMappingVectorBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorBefore,tailMappingVectorBefore)
				tempRP=2*distanceTransDL1(headVectorBefore,headMappingVectorBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorBefore,tailMappingVectorBefore)*(np.dot(headMappingVectorBefore.T,headVectorBefore)-np.dot(tailMappingVectorBefore.T,tailVectorBefore))
				tempPosT=[]
				for i in range(self.dimE):
					temp=-2*(distanceTransDL1(headVectorBefore,headMappingVectorBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorBefore,tailMappingVectorBefore)*(relationMappingVectorBefore*tailMappingVectorBefore[i]+1)).sum()
					tempPosT.append(temp)
				tempPosT=np.array(tempPosT,dtype=dType).reshape(-1,1)

				tempPosTP=[]
				for i in range(self.dimE):
					temp=-2*(distanceTransDL1(headVectorBefore,headMappingVectorBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorBefore,tailMappingVectorBefore)*(relationMappingVectorBefore*tailVectorBefore[i])).sum()
					tempPosTP.append(temp)
				tempPosTP=np.array(tempPosTP,dtype=dType).reshape(-1,1)


				tempNegH=[]
				for i in range(self.dimE):
					temp=-2*(distanceTransDL1(headVectorCorruptedBefore,headMappingVectorCorruptedBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorCorruptedBefore,tailMappingVectorCorruptedBefore)*(relationMappingVectorBefore*headMappingVectorCorruptedBefore[i]+1)).sum()
					tempNegH.append(temp)
				tempNegH=np.array(tempNegH,dtype=dType).reshape(-1,1)

				tempNegHP=[]
				for i in range(self.dimE):
					temp=-2*(distanceTransDL1(headVectorCorruptedBefore,headMappingVectorCorruptedBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorCorruptedBefore,tailMappingVectorCorruptedBefore)*(relationMappingVectorBefore*headVectorCorruptedBefore[i])).sum()
					tempNegHP.append(temp)
				tempNegHP=np.array(tempNegHP,dtype=dType).reshape(-1,1)

				tempNegT=[]
				for i in range(self.dimE):
					temp=2*(distanceTransDL1(headVectorCorruptedBefore,headMappingVectorCorruptedBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorCorruptedBefore,tailMappingVectorCorruptedBefore)*(relationMappingVectorBefore*tailMappingVectorCorruptedBefore[i]+1)).sum()
					tempNegT.append(temp)
				tempNegT=np.array(tempNegT,dtype=dType).reshape(-1,1)

				tempNegTP=[]
				for i in range(self.dimE):
					temp=2*(distanceTransDL1(headVectorCorruptedBefore,headMappingVectorCorruptedBefore,relationVectorBefore,relationMappingVectorBefore,tailVectorCorruptedBefore,tailMappingVectorCorruptedBefore)*(relationMappingVectorBefore*tailVectorCorruptedBefore[i])).sum()
					tempNegTP.append(temp)
				tempNegTP=np.array(tempNegTP,dtype=dType).reshape(-1,1)

				headVector=headVector-self.learningRate*tempPosH
				headMappingVector=headMappingVector-self.learningRate*tempPosHP
				tailVector=tailVector-self.learningRate*tempPosT
				tailMappingVector=tailMappingVector-self.learningRate*tempPosTP
				relationVector=relationVector-self.learningRate*tempR
				relationMappingVector=relationMappingVector-self.learningRate*tempRP

				headVectorCorrupted=headVectorCorrupted-self.learningRate*tempNegH
				headMappingVectorCorrupted=headMappingVectorCorrupted-self.learningRate*tempNegHP
				tailVectorCorrupted=tailVectorCorrupted-self.learningRate*tempNegT
				tailMappingVectorCorrupted=tailMappingVectorCorrupted-self.learningRate*tempNegTP

				if(np.linalg.norm(headVector)>1.0 or np.linalg.norm(headVector)<0.1):
					copyEntityList[triplets[0][0]]=norm(headVector)
				else:
					copyEntityList[triplets[0][0]]=headVector

				if(np.linalg.norm(headMappingVector)>1.0 or np.linalg.norm(headMappingVector)<0.1):
					copyEntityMappingList[triplets[0][0]]=norm(headMappingVector)
				else:
					copyEntityMappingList[triplets[0][0]]=headMappingVector

				if(np.linalg.norm(tailVector)>1.0 or np.linalg.norm(tailVector)<0.1):
					copyEntityList[triplets[0][1]]=norm(tailVector)
				else:
					copyEntityList[triplets[0][1]]=tailVector

				if(np.linalg.norm(tailMappingVector)>1.0 or np.linalg.norm(tailMappingVector)<0.1):
					copyEntityMappingList[triplets[0][1]]=norm(tailMappingVector)
				else:
					copyEntityMappingList[triplets[0][1]]=tailMappingVector

				if(np.linalg.norm(relationVector)>1.0 or np.linalg.norm(relationVector)<0.1):
					copyRelationList[triplets[0][2]]=norm(relationVector)
				else:
					copyRelationList[triplets[0][2]]=relationVector

				if(np.linalg.norm(relationMappingVector)>1.0 or np.linalg.norm(relationMappingVector)<0.1):
					copyRelationMappingList[triplets[0][2]]=norm(relationMappingVector)
				else:
					copyRelationMappingList[triplets[0][2]]=relationMappingVector

				if(np.linalg.norm(headVectorCorrupted)>1.0 or np.linalg.norm(headVectorCorrupted)<0.1):
					copyEntityList[triplets[1][0]]=norm(headVectorCorrupted)
				else:
					copyEntityList[triplets[1][0]]=headVectorCorrupted

				if(np.linalg.norm(headMappingVectorCorrupted)>1.0 or np.linalg.norm(headMappingVectorCorrupted)<0.1):
					copyEntityMappingList[triplets[1][0]]=norm(headMappingVectorCorrupted)
				else:
					copyEntityMappingList[triplets[1][0]]=headMappingVectorCorrupted

				if(np.linalg.norm(tailVectorCorrupted)>1.0 or np.linalg.norm(tailVectorCorrupted)<0.1):
					copyEntityList[triplets[1][1]]=norm(tailVectorCorrupted)
				else:
					copyEntityList[triplets[1][1]]=tailVectorCorrupted

				if(np.linalg.norm(tailMappingVectorCorrupted)>1.0 or np.linalg.norm(tailMappingVectorCorrupted)<0.1):
					copyEntityMappingList[triplets[1][1]]=norm(tailMappingVectorCorrupted)
				else:
					copyEntityMappingList[triplets[1][1]]=tailMappingVectorCorrupted

			self.entityList=copyEntityList
			self.entityMappingList=copyEntityMappingList
			self.relationList=copyRelationList
			self.relationMappingList=copyRelationMappingList
	def writeEntityVector(self,dir):
		print("writing entity")
		file=open(dir,'w')
		for entity in self.entityList.keys():
			file.write(str(entity)+'\t')
			file.write(str(self.entityList[entity].tolist())+'\t')
			file.write(str(self.entityMappingList[entity].tolist()))
			file.write('\n')
		file.close()
	def writeRelationVector(self,dir):
		file=open(dir,'w')
		for relation in self.relationList.keys():
			file.write(str(relation)+'\t')
			file.write(str(self.relationList[relation].tolist())+'\t')
			file.write(str(self.relationMappingList[relation].tolist()))
			file.write('\n')
		file.close()
			

def init(dim):
	'''
	初始化向量
	'''
	bound=6/math.sqrt(dim)
	return np.random.uniform(-bound,bound)



def distanceL2(h,l,t):
	s=h+l-t
	sum=(s*s).sum()
	return sum

def distanceTransD(h,hp,r,rp,t,tp):
	hm=np.dot(np.dot(rp,hp.T),h)
	tm=np.dot(np.dot(rp,tp.T),t)
	s=hm-tm+h+r-t
	dist=(s*s).sum()
	return dist

def distanceTransDL1(h,hp,r,rp,t,tp):
	hm=np.dot(np.dot(rp,hp.T),h)
	tm=np.dot(np.dot(rp,tp.T),t)
	s=hm-tm+h+r-t
	return s

def norm(list1):
	'''
	归一化
	'''

	if type(list1)==list:
		list1=np.array(list1,dtype=dType)

	var=np.linalg.norm(list1)

	#norm([3,4])=5
	i=0
	while i<len(list1):
		list1[i]=list1[i]/var
		i=i+1
	return np.array(list1).reshape(-1,1)
#返回行数和指定

def openDetailsAndId(dir,sp=','):
	idNum=0
	Dict={}
	with open(dir) as file:
		lines=file.readlines()
		for line in lines:
			DetailsAndId=line.strip().split(sp)
			Dict[DetailsAndId[0]]=int(DetailsAndId[1])
			idNum+=1
	return idNum,Dict
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
			triple[0]=entityDict[triple[0]]
			triple[1]=entityDict[triple[1]]
			triple[2]=relationDict[triple[2]]
			list.append(tuple(triple))
			num+=1
	return num,list

def readVectors(dir,sp='\t'):
	num=0
	dict={}
	dictp={}
	with open(dir) as file:
		lines=file.readlines()
		for line in lines:
			line=line.replace('[','').replace(']','')
			line=line.strip().split(sp)

			vector1=line[1].split(',')
			for i in range(len(vector1)):
				vector1[i]=float(vector1[i])

			vector2=line[2].split(',')
			for i in range(len(vector2)):
				vector2[i]=float(vector2[i])

			vector1=np.array(vector1,dtype=dType)
			vector2=np.array(vector2,dtype=dType)
			dict[int(line[0])]=vector1.reshape(-1,1)
			dictp[int(line[0])]=vector2.reshape(-1,1)
	return dict,dictp
if __name__ == '__main__':
	#读取数据，生成字典{'实体名':'index'}


	dirEntity="../data/WN182/entity2id.txt"
	entityNum,entityDict=openDetailsAndId(dirEntity,'\t')
	
	dirRelation = "../data/WN182/relation2id.txt"
	relationNum, relationDict = openDetailsAndId(dirRelation,'\t')
	print(entityNum,relationNum)
	dirTrain = '../data/WN182/ttt.txt'
	print("打开TransD")
	tripleNum, tripletList = openTrain(dirTrain,'\t')
	print(tripleNum)
	
	transD = TransD(entityDict,relationDict,tripletList,learningRate=0.0001 ,margin=1, dimE = 20,dimR=20)
	print("TranE初始化")
	transD.initialize()
	transD.relationList,transD.relationMappingList=readVectors("../data/WN182/relationVector.txt")
	transD.entityList,transD.entityMappingList	=readVectors("../data/WN182/entityVector.txt")

	transD.transD(cI=2000,batchNum=100)
	transD.writeEntityVector('../data/WN182/entityVector.txt')
	transD.writeRelationVector('../data/WN182/relationVector.txt')

	'''
	#transE.transE(15000)
	#transE.writeRelationVector("c:\\relationVector.txt")
	#transE.writeEntilyVector("c:\\entityVector.txt")
	'''