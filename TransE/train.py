import tensorflow as tf
import numpy as np
import csv
import math
import random
embed_dim=50
n_batch=960
margin=0.9
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
	checkpoint_dir='../data/WN18/saver/'

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
hpn=tf.reshape(tf.norm(input_h_pos,axis=1,ord=2),[-1]) #shape=[n_batch,1] - [n_batch]
#hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])
input_t_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,1]),[n_batch,embed_dim,-1])
tpn=tf.reshape(tf.norm(input_t_pos,axis=1,ord=2),[-1])
#tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])
input_r_pos=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_pos[:,2]),[n_batch,embed_dim,-1])
rpn=tf.reshape(tf.norm(input_r_pos,axis=1,ord=2),[-1])
#rp_pos=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_pos[:,2]),[n_batch,embed_dim,-1])

#mrh_pos=tf.matmul(rp_pos,hp_pos,transpose_b=True)+tf.eye(embed_dim)
#mrt_pos=tf.matmul(rp_pos,tp_pos,transpose_b=True)+tf.eye(embed_dim)
#h_pos=tf.matmul(mrh_pos,input_h_pos)
#t_pos=tf.matmul(mrt_pos,input_t_pos)

score_hrt_pos=tf.norm(input_h_pos+input_r_pos-input_t_pos,ord=1,axis=1)
#L1

#score_hrt_pos=tf.matmul((input_h_pos+input_r_pos),input_t_pos,transpose_a=True)+\
#tf.matmul(input_h_pos,(input_t_pos-input_r_pos),transpose_a=True)

train_input_neg=tf.placeholder(tf.int32,[None,3])
#(nbatch,1)
input_h_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,0]),[n_batch,embed_dim,-1])#(n_batch,1) (n_batch,dim)
hnn=tf.reshape(tf.norm(input_h_neg,axis=1,ord=2),[-1])
#hp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,0]),[n_batch,embed_dim,-1])
input_t_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,1]),[n_batch,embed_dim,-1])
tnn=tf.reshape(tf.norm(input_t_neg,axis=1,ord=2),[-1])
#tp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,1]),[n_batch,embed_dim,-1])
input_r_neg=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_neg[:,2]),[n_batch,embed_dim,-1])
rnn=tf.reshape(tf.norm(input_r_neg,axis=1,ord=2),[-1])
#rp_neg=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_neg[:,2]),[n_batch,embed_dim,-1])
#mrh_neg=tf.matmul(rp_neg,hp_neg,transpose_b=True)+tf.eye(embed_dim)
#mrt_neg=tf.matmul(rp_neg,tp_neg,transpose_b=True)+tf.eye(embed_dim)
#h_neg=tf.matmul(mrh_neg,input_h_neg)
#t_neg=tf.matmul(mrt_neg,input_t_neg)
eZeroNorm=tf.norm(tf.nn.embedding_lookup(ent_embedding,0))
rZeroNorm=tf.norm(tf.nn.embedding_lookup(rel_embedding,0))

score_hrt_neg=tf.norm(input_h_neg+input_r_neg-input_t_neg,ord=1,axis=1)
#L1
#score_hrt_neg=tf.matmul((input_h_neg+input_r_neg),input_t_neg,transpose_a=True)+\
#tf.matmul(input_h_neg,(input_t_neg-input_r_neg),transpose_a=True)

regularizer_loss=tf.reduce_sum(tf.abs(input_h_pos))+tf.reduce_sum(tf.abs(input_t_pos))+\
tf.reduce_sum(tf.abs(input_r_pos))+tf.reduce_sum(tf.abs(input_h_neg))+\
tf.reduce_sum(tf.abs(input_t_neg))+tf.reduce_sum(tf.abs(input_r_neg))

loss=tf.reduce_sum(tf.nn.relu(score_hrt_pos-score_hrt_neg+margin_))
#+regularizer_weight*regularizer_loss
optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
grads=optimizer.compute_gradients(loss,trainable)
op_train=optimizer.apply_gradients(grads)

idx_e=tf.placeholder(tf.int32,[None])
idx_r=tf.placeholder(tf.int32,[None])
normedE=tf.nn.l2_normalize(tf.nn.embedding_lookup(ent_embedding,idx_e))
normedR=tf.nn.l2_normalize(tf.nn.embedding_lookup(rel_embedding,idx_r))
updateE=tf.scatter_update(ent_embedding,idx_e,normedE)
updateR=tf.scatter_update(rel_embedding,idx_r,normedR)



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
			hp,tp,rp,hn,tn,rn,ez,rz,loss_iter,_=sess.run([hpn,tpn,\
				rpn,hnn,tnn,rnn,eZeroNorm,rZeroNorm,loss,op_train],{train_input_pos:input_pos,train_input_neg:input_neg})
			loss_sum+=loss_iter


			#print(hp)
			#hp-rn are the norms(shape:[n_batch])
			hp=(hp>1)*input_pos[:,0]
			tp=(tp>1)*input_pos[:,1]
			rp=(rp>1)*input_pos[:,2]
			hn=(hn>1)*input_neg[:,0]
			tn=(tn>1)*input_neg[:,1]
			rn=(rn>1)*input_neg[:,2]# id of the vectors which need to be normalized
			#print(hp)

			norm_elist=[]
			norm_rlist=[]
			for i in hp:
				if i!=0:
					norm_elist.append(i)
			for i in tp:
				if i!=0:
					norm_elist.append(i)
			for i in rp:
				if i!=0:
					norm_rlist.append(i)
			for i in hn:
				if i!=0:
					norm_elist.append(i)
			for i in tn:
				if i!=0:
					norm_elist.append(i)
			for i in rn:
				if i!=0:
					norm_rlist.append(i)
			if ez>1:
				norm_elist.append(0)
			if rz>1:
				norm_rlist.append(0)
			sess.run([updateE,updateR],{idx_e:norm_elist,idx_r:norm_rlist})
			if n_iter%100==0:
				print(n_iter,'/',total)
				print(loss_sum)
				loss_sum=0
				saved_path=saver.save(sess,checkpoint_dir+model_name)
	print("complete")
	saved_path=saver.save(sess,checkpoint_dir+model_name)
	print("saved\n",saved_path)

