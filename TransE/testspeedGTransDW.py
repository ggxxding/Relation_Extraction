#unif 
import tensorflow as tf
import numpy as np
import csv
import math
import random
import copy
import time
embed_dim=200
n_batch=1440
margin=0.7
weight=0.1
weight_diag=0.005
lr1=0.01
lr2=0.0001
lr=lr1
regularizer_weight=0
num_epoch=500 #500 0.01 + 500 0.0001
location='mac'
is_train=1
use_filter=0
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
	train_path='../data/FB15k/train2id.txt'
	test_path='../data/FB15k/test2id.txt'
	valid_path='../data/FB15k/valid2id.txt'
	checkpoint_dir='112/'

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
	dir='../data/FB15k/entity2id.txt'
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
	dir='../data/FB15k/relation2id.txt'
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
triplets=np.concatenate((train_triple.tolist(),test_triple.tolist(),valid_triple.tolist()),axis=0)
#triplets=triplets.tolist()
bernA=np.zeros(shape=[n_relation,n_entity,2],dtype=np.float64)

for triplet in triplets:
	bernA[triplet[2]][triplet[0]][0]+=1
	bernA[triplet[2]][triplet[1]][1]+=1
bernB=bernA>0

bern=np.sum(bernA,axis=1)/np.sum(bernB,axis=1)



def load_file(file_path):
	temp=[]
	with open(file_path,'r',encoding='utf-8') as f:
		idx=-1
		for x in f.readlines():
			if len(x.strip().split(' '))>0:
				temp.append([])
				idx+=1
				for i in x.strip().split(' '):
					temp[idx].append(int(i))
			else:
				print('length:0')
	return temp
Ehr=load_file("../data/WN18/ehr.txt")
Etr=load_file("../data/WN18/etr.txt")

filterh=load_file("../filterhWN18.txt")
filtert=load_file("../filtertWN18.txt")

'''
for i in range(n_relation):
	Ehr.append([])
	Etr.append([])
for i in train_triple:
	if i[0] not in Ehr[i[2]]:
		Ehr[i[2]].append(i[0])
	if i[1] not in Ehr[i[2]]:
		Etr[i[2]].append(i[1])

for i in test_triple:
	if i[0] not in Ehr[i[2]]:
		Ehr[i[2]].append(i[0])
	if i[1] not in Ehr[i[2]]:
		Etr[i[2]].append(i[1])

for i in valid_triple:
	if i[0] not in Ehr[i[2]]:
		Ehr[i[2]].append(i[0])
	if i[1] not in Ehr[i[2]]:
		Etr[i[2]].append(i[1])
for i in range(len(Ehr)):
	Ehr[i].sort()
	Etr[i].sort()
with open("ehr.txt",'w') as f:
	for lines in Ehr:
		for i in lines:
			f.write(str(i)+' ')
		f.write('\n')
with open("etr.txt",'w') as f:
	for lines in Etr:
		for i in lines:
			f.write(str(i)+' ')
		f.write('\n')'''

#tf.placeholder()
trainable=[] #可训练参数列表

margin_=tf.constant(margin)
weight_=tf.constant(weight)
weight_diag_=tf.constant(weight_diag)
bound = 2 / math.sqrt(embed_dim)
ent_eig=tf.get_variable("ent_eig", [n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=345))
ent_abs=tf.get_variable("ent_abs", [n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=347))
trainable.append(ent_eig)
trainable.append(ent_abs)

rel_eig=tf.get_variable("rel_eig", [n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=346))
rel_abs=tf.get_variable("rel_abs", [n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=348))
trainable.append(rel_eig)
trainable.append(rel_abs)
M_r=tf.get_variable("M_r", [14951, 1345],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=348))
#trainable.append(M_r)

'''ent_bias_t=tf.get_variable("ent_bias_t",[n_entity,embed_dim],
	initializer=tf.random_uniform_initializer(minval=-bound/5,maxval=bound/5,seed=348))
trainable.append(ent_bias_t)'''

train_input_pos=tf.placeholder(tf.int32,[None,3])

input_ha_pos=tf.reshape(tf.nn.embedding_lookup(ent_abs,train_input_pos[:,0]),[-1,embed_dim,1])#(n_batch,1) (n_batch,dim)
input_he_pos=tf.reshape(tf.nn.embedding_lookup(ent_eig,train_input_pos[:,0]),[-1,embed_dim,1])#(n_batch,1) (n_batch,dim)
#hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])
input_ta_pos=tf.reshape(tf.nn.embedding_lookup(ent_abs,train_input_pos[:,1]),[-1,embed_dim,1])
input_te_pos=tf.reshape(tf.nn.embedding_lookup(ent_eig,train_input_pos[:,1]),[-1,embed_dim,1])
#tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])
input_ra_pos=tf.reshape(tf.nn.embedding_lookup(rel_abs,train_input_pos[:,2]),[-1,embed_dim,1])
input_re_pos=tf.reshape(tf.nn.embedding_lookup(rel_eig,train_input_pos[:,2]),[-1,embed_dim,1])

Mh=tf.matmul(input_ra_pos,input_ha_pos,transpose_b=True)
Mt=tf.matmul(input_ra_pos,input_ta_pos,transpose_b=True)
hm=tf.matmul(Mh,input_he_pos)
tm=tf.matmul(Mt,input_te_pos)

h=0.5*input_he_pos+0.5*hm
t=0.5*input_te_pos+0.5*tm

score_hrt_pos1=tf.norm((h+input_re_pos-t)/2,ord=2,axis=1)
#L1

'''score_hrt_pos121=tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)
score_hrt_pos12n=0.7*tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
0.3*tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)
score_hrt_posn21=0.3*tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
0.7*tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)
score_hrt_posn2n=tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)'''
#weight_*(tf.norm(input_h_pos,ord=2)+tf.norm(input_t_pos,ord=2)+tf.norm(input_r_pos,ord=2))

train_input_neg=tf.placeholder(tf.int32,[None,3])

input_ha_neg=tf.reshape(tf.nn.embedding_lookup(ent_abs,train_input_neg[:,0]),[-1,embed_dim,1])#(n_batch,1) (n_batch,dim)
input_he_neg=tf.reshape(tf.nn.embedding_lookup(ent_eig,train_input_neg[:,0]),[-1,embed_dim,1])#(n_batch,1) (n_batch,dim)
#hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])
input_ta_neg=tf.reshape(tf.nn.embedding_lookup(ent_abs,train_input_neg[:,1]),[-1,embed_dim,1])
input_te_neg=tf.reshape(tf.nn.embedding_lookup(ent_eig,train_input_neg[:,1]),[-1,embed_dim,1])
#tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])
input_ra_neg=tf.reshape(tf.nn.embedding_lookup(rel_abs,train_input_neg[:,2]),[-1,embed_dim,1])
input_re_neg=tf.reshape(tf.nn.embedding_lookup(rel_eig,train_input_neg[:,2]),[-1,embed_dim,1])

Mh=tf.matmul(input_ra_neg,input_ha_neg,transpose_b=True)
Mt=tf.matmul(input_ra_neg,input_ta_neg,transpose_b=True)
hm=tf.matmul(Mh,input_he_neg)
tm=tf.matmul(Mt,input_te_neg)

h=0.5*input_he_neg+0.5*hm
t=0.5*input_te_neg+0.5*tm

score_hrt_neg1=tf.norm((h+input_re_neg-t)/2,ord=2,axis=1)
#L1
'''score_hrt_neg121=tf.reduce_sum((input_h_neg+input_r_neg)*input_t_neg,axis=1)+\
tf.reduce_sum(input_h_neg*(input_t_neg-input_r_neg),axis=1)
score_hrt_neg12n=0.7*tf.reduce_sum((input_h_neg+input_r_neg)*input_t_neg,axis=1)+\
0.3*tf.reduce_sum(input_h_neg*(input_t_neg-input_r_neg),axis=1)

score_hrt_negn21=0.3*tf.reduce_sum((input_h_neg+input_r_neg)*input_t_neg,axis=1)+\
0.7*tf.reduce_sum(input_h_neg*(input_t_neg-input_r_neg),axis=1)
score_hrt_negn2n=tf.reduce_sum((input_h_neg+input_r_neg)*input_t_neg,axis=1)+\
tf.reduce_sum(input_h_neg*(input_t_neg-input_r_neg),axis=1)'''
#weight_*(tf.norm(input_h_neg,ord=2)+tf.norm(input_t_neg,ord=2)+tf.norm(input_r_neg,ord=2))


loss=tf.reduce_sum(tf.nn.relu(margin_+score_hrt_pos1-score_hrt_neg1))
'''+weight_*\
(tf.reduce_sum(tf.square(input_h_pos))+tf.reduce_sum(tf.square(input_r_pos))+\
	tf.reduce_sum(tf.square(input_t_pos))+tf.reduce_sum(tf.square(input_h_neg))+\
	tf.reduce_sum(tf.square(input_r_neg))+tf.reduce_sum(tf.square(input_t_neg))+\
	tf.reduce_sum(tf.square(bias_h_pos))+tf.reduce_sum(tf.square(bias_t_pos))+\
	tf.reduce_sum(tf.square(bias_h_neg))+tf.reduce_sum(tf.square(bias_t_neg)))'''

#weight_diag_*(tf.reduce_sum(tf.square(diag_pos))+tf.reduce_sum(tf.square(diag_neg)))
#+regularizer_weight*regularizer_loss
optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
#optimizer=tf.train.AdadeltaOptimizer()
grads=optimizer.compute_gradients(loss,trainable)
#grads=optimizer.compute_gradients(loss)
op_train=optimizer.apply_gradients(grads)


saver=tf.train.Saver()



# 启动图 (graph)
init=tf.global_variables_initializer()
init_local_op=tf.initialize_local_variables()
loss_sum=0
with tf.Session() as sess:
	if is_train==1:
		n_triple=train_triple.shape[0]
		print("train triplets:%d"%(n_triple))
		sess.run(init)
		sess.run(init_local_op)
		'''ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			print("success! %s."% ckpt.model_checkpoint_path)
			saver.restore(sess,ckpt.model_checkpoint_path)
		else:
			print('fail to restore')'''
		n_epoch=0
		n_iter=0
		total=math.ceil(n_triple/n_batch)*num_epoch*2
		start=time.perf_counter()
		print(start)
		while n_epoch<num_epoch*2:
			n_epoch+=1
			if n_epoch>num_epoch:
				lr=lr2
			n_idx=0
			while(n_idx<n_triple):
				n_iter+=1
				if n_idx+n_batch>n_triple:
					input_pos=np.concatenate([train_triple[n_idx:n_idx+n_batch],train_triple[0:n_idx+n_batch-n_triple]],axis=0)
					'''train_triple=train_triple.tolist()
					random.shuffle(train_triple)
					train_triple=np.asarray(train_triple,dtype=np.int32)
					print("shuffled")'''
				else:
					input_pos=train_triple[n_idx:n_idx+n_batch]
				#input_pos=np.asarray(input_pos,dtype=np.int32)
				n_idx+=n_batch
				temp=input_pos.tolist()
				input_neg=[]
				for idx in range(input_pos.shape[0]):
					if np.random.uniform(-1,1) > 0:
					#if np.random.uniform(0,1) > bern[input_pos[idx][2]][0]/(bern[input_pos[idx][2]][0]+bern[input_pos[idx][2]][1]):
						temp_ent=random.sample(entityID_list,1)[0]			
						input_neg.append([temp[idx][0],int(temp_ent),temp[idx][2]])
						'''temp_ent=random.sample(Ehr[input_pos[idx][2]],1)[0]
						input_neg.append([temp_ent,temp[idx][1],temp[idx][2]])'''
					else:
						temp_ent=random.sample(entityID_list,1)[0]	
						input_neg.append([int(temp_ent),temp[idx][1],temp[idx][2]])
						'''temp_ent=random.sample(Etr[input_pos[idx][2]],1)[0]
						input_neg.append([temp[idx][0],temp_ent,temp[idx][2]])'''
				input_neg=np.asarray(input_neg,dtype=np.int32)
				for i in range(input_pos.shape[0]):
					a=M_r
					#a=a[[input_pos[i,0],input_pos[i,2]]]
				#print(input_neg)
				loss_iter,_=sess.run([loss,op_train],\
					{train_input_pos:input_pos,train_input_neg:input_neg})
				loss_sum+=loss_iter
				if n_iter>5000:
					break
				if n_iter%1000==0:
					dur=time.perf_counter()
					#print(n_iter,'/',total,' learning rate:',lr)
					print(n_iter,'/',total)
					#print(dur)
					print(dur-start)
					loss_sum=0
					#saved_path=saver.save(sess,checkpoint_dir+model_name)
		print("complete")
		'''saved_path=saver.save(sess,checkpoint_dir+model_name)
		print("saved\n",saved_path)'''
	elif is_train==0:
		n_triple=test_triple.shape[0]
		print("test triplets:%d"%(n_triple))
		n_iter=0
		ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			print("success! %s."% ckpt.model_checkpoint_path)
			saver.restore(sess,ckpt.model_checkpoint_path)
		else:
			print('fail to restore')
		n_epoch=0
		rank_list=[]

		while n_epoch<1:
			n_epoch+=1
			n_idx=0
			while(n_idx<n_triple):#n_triple=n_test_triple
				input_pos=test_triple[n_idx:n_idx+1]
				n_idx+=1
				#temp_input=input_pos.tolist()
				#head
				index=input_pos[0][0]
				input_list=[]
				#temp_idx=0
				count=0
				for idx in range(n_entity):
					if use_filter==1:
						if idx not in filterh[n_idx-1]:
							input_list.append([idx,input_pos[0][1],input_pos[0][2]])
							count+=1
						elif idx==index:
							input_list.append([idx,input_pos[0][1],input_pos[0][2]])
							index=count
					else:
						input_list.append([idx,input_pos[0][1],input_pos[0][2]])

				scores=sess.run(score_hrt_pos1,{train_input_pos:input_list})
				scores=scores.reshape(-1).tolist()
				#index=temp_idx
				temp=scores[index]
				#temp=scores[temp_idx]
				scores.sort(reverse=False)
				index_=scores.index(temp)
				rank_list.append(index_)
				print(index_)

				#tail
				index=input_pos[0][1]
				input_list=[]
				count=0
				#temp_idx=0
				for idx in range(n_entity):
					if use_filter==1:
						if idx not in filtert[n_idx-1]:
							input_list.append([input_pos[0][0],idx,input_pos[0][2]])
							count+=1
						elif idx==index:
							input_list.append([input_pos[0][0],idx,input_pos[0][2]])
							index=count
					else:
						input_list.append([input_pos[0][0],idx,input_pos[0][2]])
				scores=sess.run(score_hrt_pos1,{train_input_pos:input_list})
				#scores=sess.run(score_hrt_pos,{train_input_pos:input_list})
				scores=scores.reshape(-1).tolist()
				#index=temp_idx
				#temp=scores[temp_idx]
				temp=scores[index]
				scores.sort(reverse=False)
				index_=scores.index(temp)
				rank_list.append(index_)

				print(index_)


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