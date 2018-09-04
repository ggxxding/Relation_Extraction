import numpy as np
import random#random.sample
import math
from copy import deepcopy
import csv
dType='float64'
def init(dim):
	'''
	初始化向量
	'''

	return np.random.uniform(-1,1)


def distanceL2(h,l,t):
	s=h+l-t
	sum=(s*s).sum()
	return sum

def distanceTransD(h,hp,r,rp,t,tp):
	hm=np.dot(np.dot(rp,hp.T),h)
	tm=np.dot(np.dot(rp,tp.T),t)
	s=hm-tm+r
	dist=(s*s).sum()
	return dist

def distanceTransDL1(h,hp,r,rp,t,tp):
	hm=np.dot(np.dot(rp,hp.T),h)
	tm=np.dot(np.dot(rp,tp.T),t)
	s=norm(hm)-norm(tm)+norm(r)
	return s

def norm(list1):
	'''
	归一化
	'''
	list1=np.array(list1,dtype='float32')##
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
			dict[int(line[0])]=[vector1,vector2]
	return dict



if __name__ == '__main__':
	#读取数据，生成字典{'实体名':'index'}
	dirEntity="../data/WN182/entity2id.txt"
	entityNum,entityDict=openDetailsAndId(dirEntity,'\t')
	
	dirRelation = "../data/WN182/relation2id.txt"
	relationNum, relationDict = openDetailsAndId(dirRelation,'\t')

	relationVectorDict=readVectors("../data/WN182/relationVector.txt")
	entityVectorDict=readVectors("../data/WN182/entityVector.txt")

	dirTrain = '../data/WN182/ttt.txt'
	print("打开TransD")
	tripleNum, tripletList = openTrain(dirTrain,'\t')
	'''#bern
	relationAttr={}#{int(index):[hpt,tph]}
	temp=[0,0]
	numH=0
	numT=0
	flagH=0
	flagT=0
	for r in relationVectorDict:
		for e in entityVectorDict:
			for triplet in tripletList:
				if (e,r)==(triplet[0],triplet[2]):
					temp[0]+=1
					flagH=1
				if (e,r)==(triplet[1],triplet[2]):
					temp[1]+=1
					flagT=1
			if flagH==1:
				numH+=1
				flagH=0
			if flagT==1:
				numT+=1
				flagT=0
		if numH!=0:
			temp[0]=temp[0]/numH
		if numT!=0:
			temp[1]=temp[1]/numT
		print(temp,numH,numT)
		relationAttr[r]=temp
		temp=[0,0]
		numH=0
		numT=0
	print(relationAttr)
	#bern#'''
	for i in tripletList:
		print(relationVectorDict[i[2]])


	'''
	#transE.transE(15000)
	#transE.writeRelationVector("c:\\relationVector.txt")
	#transE.writeEntilyVector("c:\\entityVector.txt")
	'''