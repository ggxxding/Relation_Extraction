import numpy as np
import random#random.sample
import math
from copy import deepcopy
import csv

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

if __name__ == '__main__':
	#读取数据，生成字典{'实体名':'index'}


	dirEntity="../data/WN182/entity2id.txt"
	entityNum,entityDict=openDetailsAndId(dirEntity,'\t')
	
	dirRelation = "../data/WN182/relation2id.txt"
	relationNum, relationDict = openDetailsAndId(dirRelation,'\t')

	dirTrain = '../data/WN182/ttt.txt'
	print("打开TransD")
	tripleNum, tripletList = openTrain(dirTrain,'\t')

	print(tripletList)
	'''
	#transE.transE(15000)
	#transE.writeRelationVector("c:\\relationVector.txt")
	#transE.writeEntilyVector("c:\\entityVector.txt")
	'''