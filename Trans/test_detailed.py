import tensorflow as tf
import numpy as np
import csv
import math
import random
embed_dim=50
n_batch=1	#needn't change it
num_epoch=1	#needn't change it
if location=='104':
	train_path='/data/Relation_Extraction/data/WN18/train.txt'
	test_path='/data/Relation_Extraction/data/WN18/test.txt'
	valid_path='/data/Relation_Extraction/data/WN18/valid.txt'
	checkpoint_dir='/data/Relation_Extraction/data/WN18/saver/'
elif location=='local':
	train_path='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/train.txt'
	test_path='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/test.txt'
	valid_path='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/valid.txt'
	checkpoint_dir='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/saver/'
elif location=='mac':
	train_path='../data/WN182/train.txt'
	test_path='../data/WN182/test.txt'
	valid_path='../data/WN182/valid.txt'
	checkpoint_dir='../data/WN182/saver/'

model_name='modeld'
entity_id_map={}
id_entity_map={}
relation_id_map={}
id_relation_map={}
if location=='104':
	dir='/data/Relation_Extraction/data/WN18/entity2id.txt'
elif location=='local':
	dir='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/entity2id.txt'
elif location=='mac':
	dir='../data/WN182/entity2id.txt'
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
	dir='../data/WN182/relation2id.txt'
csv_file=csv.reader(open(dir))
n_relation=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_relation+=1
	relation_id_map[line[0]]=line[1]
	#id_relation_map[line[1]]=line[0]
print("entity number:%d,relation number:%d"%(n_entity,n_relation))
entityID_list=list(entity_id_map.values())
relationID_list=list(relation_id_map.values())

entity_list=list(entity_id_map.keys())
relation_list=list(relation_id_map.keys())

def load_triple(file_path):
	with open(file_path,'r',encoding='utf-8') as f_triple:
		return np.asarray([[entity_id_map[x.strip().split('\t')[0]],
			entity_id_map[x.strip().split('\t')[1]],
			relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
			dtype=np.int32)
test_triple=load_triple(test_path)


n_triple=test_triple.shape[0]
print("test triplets:%d"%(n_triple))

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
input_h_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,0]),[n_entity,embed_dim,-1])#(n_batch,1) (n_batch,dim)
#hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])
input_t_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,1]),[n_entity,embed_dim,-1])
#tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])
input_r_pos=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_pos[:,2]),[n_entity,embed_dim,-1])
#rp_pos=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_pos[:,2]),[n_batch,embed_dim,-1])

#mrh_pos=tf.matmul(rp_pos,hp_pos,transpose_b=True)+tf.eye(embed_dim)
#mrt_pos=tf.matmul(rp_pos,tp_pos,transpose_b=True)+tf.eye(embed_dim)
#h_pos=tf.matmul(mrh_pos,input_h_pos)
#t_pos=tf.matmul(mrt_pos,input_t_pos)
score_hrt_pos=tf.norm(input_h_pos+input_r_pos-input_t_pos,ord=1,axis=1)
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

		while(n_idx<n_triple):
			input_pos=test_triple[n_idx:n_idx+1]
			n_idx+=1
			temp=input_pos.tolist()
			#head
			index=input_pos[0][0]
			input_list=[]
			for idx in range(n_entity):
				input_list.append([idx,temp[0][1],temp[0][2]])
			scores=sess.run(score_hrt_pos,{train_input_pos:input_list})
			scores=scores.reshape(-1).tolist()
			temp=scores[index]
			scores.sort()
			rank_list.append(scores.index(temp))
			print(scores.index(temp))

			#tail
			index=input_pos[0][1]
			input_list=[]
			for idx in range(n_entity):
				input_list.append([temp[0][0],idx,temp[0][2]])
			scores=sess.run(score_hrt_pos,{train_input_pos:input_list})
			scores=scores.reshape(-1).tolist()
			temp=scores[index]
			scores.sort()
			rank_list.append(scores.index(temp))
			print(scores.index(temp))

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

