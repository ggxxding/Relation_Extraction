import tensorflow as tf
import numpy as np
import csv
import math
import random
import copy
embed_dim=50
n_batch=1	#needn't change it
num_epoch=1	#needn't change it
location='mac'
if location=='104':
	test_path='/data/Relation_Extraction/data/WN18/test.txt'
	checkpoint_dir='/data/Relation_Extraction/data/WN18/saver/'
elif location=='local':
	test_path='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/test.txt'
	checkpoint_dir='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/saver/'
elif location=='mac':
	train_path='../data/WN18/train2id.txt'
	test_path='../data/WN18/test2id.txt'
	valid_path='../data/WN18/valid2id.txt'
	checkpoint_dir='e50b960m0.9WN18/'
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
	dir='../data/WN18/entity2id.txt'
csv_file=csv.reader(open(dir))
n_entity=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_entity+=1
	entity_id_map[line[0]]=line[1]
	id_entity_map[line[1]]=line[0]

if location=='104':
	dir='/data/Relation_Extraction/data/WN18/relation2id.txt'
elif location=='local':
	dir='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/relation2id.txt'
elif location=='mac':
	dir='../data/WN18/relation2id.txt'
csv_file=csv.reader(open(dir))
n_relation=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_relation+=1
	relation_id_map[line[0]]=line[1]
	id_relation_map[line[1]]=line[0]
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
triplets=np.concatenate((train_triple.tolist(),test_triple.tolist(),valid_triple.tolist()),axis=0)
triplets=triplets.tolist()


n_triple=test_triple.shape[0]
print("test triplets:%d"%(n_triple))

filterh=[]
filtert=[]
with open("../filterhWN18.txt",'r') as f:
	idx=-1
	for x in f.readlines():
		if len(x.strip().split(' '))>0:
			filterh.append([])
			idx+=1
			for i in x.strip().split(' '):
				filterh[idx].append(int(i))
		else:
			print('length:0')
with open("../filtertWN18.txt",'r') as f:
	idx=-1
	for x in f.readlines():
		if len(x.strip().split(' '))>0:
			filtert.append([])
			idx+=1
			for i in x.strip().split(' '):
				filtert[idx].append(int(i))
		else:
			print('length:0')
#tf.placeholder()
trainable=[] #可训练参数列表


bound = 6 / math.sqrt(embed_dim)
ent_embedding =tf.get_variable("ent_embedding", [n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=345))
'''ent_projecting=tf.get_variable("ent_projecting", [n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=347))'''
trainable.append(ent_embedding)
#trainable.append(ent_projecting)

rel_embedding =tf.get_variable("rel_embedding", [n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=346))
'''rel_projecting=tf.get_variable("rel_projecting", [n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=348))'''
trainable.append(rel_embedding)
#trainable.append(rel_projecting)

train_input_pos=tf.placeholder(tf.int32,[None,3])
#(nbatch,1)
input_h_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,0]),[-1,embed_dim,1])#(n_batch,1) (n_batch,dim)
#hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])
input_t_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,1]),[-1,embed_dim,1])
#tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])
input_r_pos=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_pos[:,2]),[-1,embed_dim,1])
#rp_pos=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_pos[:,2]),[n_batch,embed_dim,-1])

#mrh_pos=tf.matmul(rp_pos,hp_pos,transpose_b=True)+tf.eye(embed_dim)
#mrt_pos=tf.matmul(rp_pos,tp_pos,transpose_b=True)+tf.eye(embed_dim)
#h_pos=tf.matmul(mrh_pos,input_h_pos)
#t_pos=tf.matmul(mrt_pos,input_t_pos)
#score_hrt_pos=tf.norm(input_h_pos+input_r_pos-input_t_pos,ord=1,axis=1)
#L1

score_hrt_pos1=tf.norm(input_h_pos+input_r_pos-input_t_pos,ord=1,axis=1)
#L1

score_hrt_pos=tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)-score_hrt_pos1

train_input_neg=tf.placeholder(tf.int32,[None,3])
#(nbatch,1)
input_h_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,0]),[n_entity,embed_dim,-1])#(n_batch,1) (n_batch,dim)
#hp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,0]),[n_batch,embed_dim,-1])
input_t_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,1]),[n_entity,embed_dim,-1])
#tp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,1]),[n_batch,embed_dim,-1])
input_r_neg=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_neg[:,2]),[n_entity,embed_dim,-1])
#rp_neg=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_neg[:,2]),[n_batch,embed_dim,-1])
#mrh_neg=tf.matmul(rp_neg,hp_neg,transpose_b=True)+tf.eye(embed_dim)
#mrt_neg=tf.matmul(rp_neg,tp_neg,transpose_b=True)+tf.eye(embed_dim)
#h_neg=tf.matmul(mrh_neg,input_h_neg)
#t_neg=tf.matmul(mrt_neg,input_t_neg)
score_hrt_neg=tf.norm(input_h_neg+input_r_neg-input_t_neg,ord=1,axis=1)

saver=tf.train.Saver()

init=tf.global_variables_initializer()
init_local_op=tf.initialize_local_variables()
with tf.Session() as sess:
	n_iter=0
	#sess.run(init)
	#sess.run(init_local_op)
	ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		print("success! %s."% ckpt.model_checkpoint_path)
		saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		print('fail to restore')

	n_epoch=0
	rank_list=[]
	while n_epoch<num_epoch:
		n_epoch+=1
		n_idx=0
		h_count=0
		t_count=0
		while(n_idx<n_triple):
			input_pos=test_triple[n_idx:n_idx+1]
			n_idx+=1
			#temp_input=input_pos.tolist()
			#head
			index=input_pos[0][0]
			input_list=[]
			temp_idx=0
			count=0
			for idx in range(n_entity):
				if idx not in filterh[n_idx-1]:
					input_list.append([idx,input_pos[0][1],input_pos[0][2]])
					count+=1
				elif idx==index:
					input_list.append([idx,input_pos[0][1],input_pos[0][2]])
					temp_idx=count
			scores=sess.run(score_hrt_pos,{train_input_pos:input_list})
			scores=scores.reshape(-1).tolist()
			temp=scores[temp_idx]
			scores_unsort=copy.deepcopy(scores)
			scores.sort(reverse=True)
			rank_list.append(scores.index(temp))
			#print('h',scores.index(temp))

			if scores.index(temp)<10:
				h_count+=1
				temp_trip=[]

				trip_list=[]
				for i in scores[:10]:
					trip_list.append(scores_unsort.index(i))
				print(trip_list)

				
				for i in  trip_list:
					temp_trip.append(id_entity_map[str(input_list[i][0])]
						)
				print(id_entity_map[str(input_list[i][1])],id_relation_map[str(input_list[i][2])])
				if h_count<10:
					print(temp_trip,scores.index(temp),'h')

			#tail
			
			index=input_pos[0][1]
			input_list=[]
			count=0
			temp_idx=0
			for idx in range(n_entity):
				if idx not in filtert[n_idx-1]:
					input_list.append([input_pos[0][0],idx,input_pos[0][2]])
					count+=1
				elif idx==index:
					input_list.append([input_pos[0][0],idx,input_pos[0][2]])
					temp_idx=count

			scores=sess.run(score_hrt_pos,{train_input_pos:input_list})
			scores=scores.reshape(-1).tolist()
			temp=scores[temp_idx]
			scores_unsort=copy.deepcopy(scores)
			scores.sort(reverse=True)
			rank_list.append(scores.index(temp))
			#print('t',scores.index(temp))

			if scores.index(temp)<10:
				t_count+=1

				temp_trip=[]
				trip_list=[]
				for i in scores[:10]:
					trip_list.append(scores_unsort.index(i))
				print(trip_list)
				for i in  trip_list:
					temp_trip.append(
						id_entity_map[str(input_list[i][1])]
						)
				print(id_entity_map[str(input_list[i][0])],id_relation_map[str(input_list[i][2])])
				if t_count<10:
					print(temp_trip,scores.index(temp),'t')
					
			if h_count>4 and t_count>4:
				break

			if n_idx%100==0:
				print(n_idx,'/',n_triple)


	hits10=0
	for i in rank_list:
		if i<=10:
			hits10+=1
	hits10=hits10/len(rank_list)
	rank_list=np.asarray(rank_list,dtype=np.int32)
	mean_rank=np.sum(rank_list)/rank_list.shape[0]
	print('hits10:',hits10)
	print('meanrank:',mean_rank)
	print("completed")

