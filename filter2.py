import tensorflow as tf
import numpy as np
import csv
import math
import random
embed_dim=50
n_batch=200
margin=2.0
lr1=0.01
lr2=0.0001
lr=lr1
regularizer_weight=0
num_epoch=400 #500 0.01 + 500 0.0001
location='mac'
#n_entity/n_relation/n_triple
#dict_type:  str:str
#train/test_triple_type: [[int32,int32,int32]...]

if location=='104':
	train_path='/data/Relation_Extraction/data/WN18/train.txt'
	checkpoint_dir='/data/Relation_Extraction/data/WN18/saver/'
elif location=='local':
	train_path='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/train.txt'
	checkpoint_dir='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/saver/'
elif location=='mac':
	train_path='data/FB15k/train2id.txt'
	test_path='data/FB15k/test2id.txt'
	valid_path='data/FB15k/valid2id.txt'
	relation_path='data/FB15k/1-1.txt'


model_name='modele'
entity_id_map={}
id_entity_map={}
relation_id_map={}
id_relation_map={}
if location=='104':
	dir='/data/Relation_Extraction/data/WN18/entity2id.txt'
elif location=='local':
	dir='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/entity2id.txt'
elif location=='mac':
	dir='data/FB15k/entity2id.txt'
csv_file=csv.reader(open(dir))
n_entity=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_entity+=1
	entity_id_map[line[0]]=line[1]
	#id_entity_map[line[1]]=line[0]
if location=='104':
	dir='/data/Relation_Extraction/data/WN18/relation2id.txt'
elif location=='local':
	dir='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/relation2id.txt'
elif location=='mac':
	dir='data/FB15k/relation2id.txt'
csv_file=csv.reader(open(dir))
n_relation=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_relation+=1
	relation_id_map[line[0]]=line[1]
	#id_relation_map[line[1]]=line[0]
print("entity number:%d,relation number:%d"%(n_entity,n_relation))
#print(entity_id_map)
entityID_list=list(entity_id_map.values())
relationID_list=list(relation_id_map.values())
def load_triple_file(file_path):
	with open(file_path,'r',encoding='utf-8') as f_triple:
		return np.asarray([[entity_id_map[x.strip().split('\t')[0]],
			entity_id_map[x.strip().split('\t')[1]],
			relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
			dtype=np.int32)

def load_triple(file_path):
	temp=[]
	with open(file_path,'r',encoding='utf-8') as f_triple:
		for x in f_triple.readlines():
			if len(x.strip().split(' '))==3:
				temp.append([x.strip().split(' ')[0],
					x.strip().split(' ')[1],
					x.strip().split(' ')[2]])
		return np.asarray(temp,dtype=np.int32)

train_triple=load_triple(train_path)
test_triple=load_triple(test_path)
valid_triple=load_triple(valid_path)
test2_triple=load_triple(relation_path)
triplets=np.concatenate((train_triple.tolist(),test_triple.tolist(),valid_triple.tolist()),axis=0)
#triplets=triplets.tolist()
filterh=[]
filtert=[]
idx=-1
for i in test2_triple:
	filterh.append([])
	filtert.append([])
	idx+=1
	for j in triplets:

		if ([i[1],i[2]]==[j[1],j[2]]):
			filterh[idx].append(j[0])
		if ([i[0],i[2]]==[j[0],j[2]]):
			filtert[idx].append(j[1])

	print(len(filterh[idx]),len(filtert[idx]))
	print(idx)
	with open("filterhFB15k1-1.txt",'a+') as f:
		for i in filterh[idx]:
			f.write(str(i)+' ')
		f.write('\n')

	with open("filtertFB15k1-1.txt",'a+') as f:
		for i in filtert[idx]:
			f.write(str(i)+' ')
		f.write('\n')
	print('writed')



print(len(filterh),len(filtert))