import tensorflow as tf
import numpy as np
import csv
import math
import random
embed_dim=50
n_batch=512
margin=2.
lr=0.0001
regularizer_weight=0
num_epoch=500
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
	train_path='../data/WN18/train.txt'
	checkpoint_dir='../data/WN18/saver3/'

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
	#id_entity_map[line[1]]=line[0]
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
	#id_relation_map[line[1]]=line[0]
print("entity number:%d,relation number:%d"%(n_entity,n_relation))
#print(entity_id_map)
entityID_list=list(entity_id_map.values())
relationID_list=list(relation_id_map.values())

def load_triple(file_path):
	with open(file_path,'r',encoding='utf-8') as f_triple:
		return np.asarray([[entity_id_map[x.strip().split('\t')[0]],
			entity_id_map[x.strip().split('\t')[1]],
			relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
			dtype=np.int32)
train_triple=load_triple(train_path)

n_triple=train_triple.shape[0]
print("train triplets:%d"%(n_triple))

#tf.placeholder()
trainable=[] #可训练参数列表

margin_=tf.constant(margin)
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

input_h_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,0]),[n_batch,embed_dim,-1])#(n_batch,1) (n_batch,dim)
hpn=tf.reshape(tf.norm(input_h_pos,axis=1,ord=2),[n_batch,1,1])
hpn_=tf.tile(hpn,[1,embed_dim,1])
hpn__=tf.where(hpn_>1,tf.nn.l2_normalize(input_h_pos),input_h_pos)
#hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])
input_t_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,1]),[n_batch,embed_dim,-1])
tpn=tf.reshape(tf.norm(input_t_pos,axis=1,ord=2),[n_batch,1,1])
tpn_=tf.tile(tpn,[1,embed_dim,1])
tpn__=tf.where(tpn_>1,tf.nn.l2_normalize(input_t_pos),input_t_pos)
#tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])
input_r_pos=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_pos[:,2]),[n_batch,embed_dim,-1])
rpn=tf.reshape(tf.norm(input_r_pos,axis=1,ord=2),[n_batch,1,1])
rpn_=tf.tile(rpn,[1,embed_dim,1])
rpn__=tf.where(rpn_>1,tf.nn.l2_normalize(input_r_pos),input_r_pos)

score_hrt_pos=tf.norm(hpn__+rpn__-tpn__,ord=1,axis=1)
#L1

train_input_neg=tf.placeholder(tf.int32,[None,3])
#(nbatch,1)
input_h_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,0]),[n_batch,embed_dim,-1])#(n_batch,1) (n_batch,dim)
hnn=tf.reshape(tf.norm(input_h_neg,axis=1,ord=2),[n_batch,1,1])
hnn_=tf.tile(hnn,[1,embed_dim,1])
hnn__=tf.where(hnn_>1,tf.nn.l2_normalize(input_h_neg),input_h_neg)
#hp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,0]),[n_batch,embed_dim,-1])
input_t_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,1]),[n_batch,embed_dim,-1])
tnn=tf.reshape(tf.norm(input_t_neg,axis=1,ord=2),[n_batch,1,1])
tnn_=tf.tile(tnn,[1,embed_dim,1])
tnn__=tf.where(tnn_>1,tf.nn.l2_normalize(input_t_neg),input_t_neg)
#tp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,1]),[n_batch,embed_dim,-1])
input_r_neg=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_neg[:,2]),[n_batch,embed_dim,-1])
rnn=tf.reshape(tf.norm(input_r_neg,axis=1,ord=2),[n_batch,1,1])
rnn_=tf.tile(rnn,[1,embed_dim,1])
rnn__=tf.where(rnn_>1,tf.nn.l2_normalize(input_r_neg),input_r_neg)
#rp_neg=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_neg[:,2]),[n_batch,embed_dim,-1])
#mrh_neg=tf.matmul(rp_neg,hp_neg,transpose_b=True)+tf.eye(embed_dim)
#mrt_neg=tf.matmul(rp_neg,tp_neg,transpose_b=True)+tf.eye(embed_dim)
#h_neg=tf.matmul(mrh_neg,input_h_neg)
#t_neg=tf.matmul(mrt_neg,input_t_neg)
score_hrt_neg=tf.norm(hnn__+rnn__-tnn__,ord=1,axis=1)
#L1

loss=tf.reduce_sum(tf.nn.relu(score_hrt_pos-score_hrt_neg+margin_))
#+regularizer_weight*regularizer_loss
optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
grads=optimizer.compute_gradients(loss)
op_train=optimizer.apply_gradients(grads)



saver=tf.train.Saver()



# 启动图 (graph)
init=tf.global_variables_initializer()
init_local_op=tf.initialize_local_variables()
loss_sum=0
with tf.Session() as sess:

	sess.run(init)
	sess.run(init_local_op)
	ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		print("success! %s."% ckpt.model_checkpoint_path)
		saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		print('fail to restore')
	n_epoch=0
	n_iter=0
	total=math.ceil(n_triple/n_batch)*num_epoch
	while n_epoch<num_epoch:
		n_epoch+=1
		n_idx=0
		while(n_idx<n_triple):
			n_iter+=1
			if n_idx+n_batch>n_triple:
				input_pos=np.concatenate([train_triple[n_idx:n_idx+n_batch],train_triple[0:n_idx+n_batch-n_triple]],axis=0)
			else:
				input_pos=train_triple[n_idx:n_idx+n_batch]
			#input_pos=np.asarray(input_pos,dtype=np.int32)

			n_idx+=n_batch
			temp=input_pos.tolist()
			input_neg=[]
			for idx in range(input_pos.shape[0]):
				if np.random.uniform(-1,1) > 0:
					temp_ent=random.sample(entityID_list,1)[0]			
					input_neg.append([int(temp_ent),temp[idx][1],temp[idx][2]])
				else:
					temp_ent=random.sample(entityID_list,1)[0]	
					input_neg.append([temp[idx][0],int(temp_ent),temp[idx][2]])
			input_neg=np.asarray(input_neg,dtype=np.int32)
			#print(input_neg)
			loss_iter,_=sess.run([loss,op_train],{train_input_pos:input_pos,train_input_neg:input_neg})
			loss_sum+=loss_iter



			if n_iter%100==0:
				print(n_iter,'/',total)
				print(loss_sum)
				loss_sum=0
				saved_path=saver.save(sess,checkpoint_dir+model_name)
	print("complete")
	saved_path=saver.save(sess,checkpoint_dir+model_name)
	print("saved\n",saved_path)

