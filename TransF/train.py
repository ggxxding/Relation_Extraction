import tensorflow as tf
import numpy as np
import csv
import math
import random
import copy
embed_dim=50
n_batch=960
margin=0.9
weight=0.2
weight_diag=0.005
lr1=0.01
lr2=0.0001
lr=lr1
regularizer_weight=0
num_epoch=500 #500 0.01 + 500 0.0001
location='mac'
is_train=0
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
	train_path='../data/WN18/train2id.txt'
	test_path='../data/WN18/test2id.txt'
	valid_path='../data/WN18/valid2id.txt'
	checkpoint_dir='e50b960m1.5w1.5WN18/'

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
diag_matrix=tf.get_variable("diag_matrix",[n_relation,embed_dim],
	initializer=tf.random_uniform_initializer(minval=0.9,maxval=1.1,seed=347))
trainable.append(diag_matrix)
train_input_pos=tf.placeholder(tf.int32,[None,3])
#(nbatch,1)

input_h_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,0]),[-1,embed_dim,1])#(n_batch,1) (n_batch,dim)
hpn=tf.reshape(tf.norm(input_h_pos,axis=1,ord=2),[-1]) #shape=[n_batch,1] - [n_batch]
#hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])
input_t_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,1]),[-1,embed_dim,1])
tpn=tf.reshape(tf.norm(input_t_pos,axis=1,ord=2),[-1])
#tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])
input_r_pos=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_pos[:,2]),[-1,embed_dim,1])
rpn=tf.reshape(tf.norm(input_r_pos,axis=1,ord=2),[-1])
#rp_pos=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_pos[:,2]),[n_batch,embed_dim,-1])
diag_pos=tf.abs(tf.reshape(tf.nn.embedding_lookup(diag_matrix,train_input_pos[:,2]),[-1,embed_dim,1]))
#shape =[batch,dim,1]
#mrh_pos=tf.matmul(rp_pos,hp_pos,transpose_b=True)+tf.eye(embed_dim)
#mrt_pos=tf.matmul(rp_pos,tp_pos,transpose_b=True)+tf.eye(embed_dim)
#h_pos=tf.matmul(mrh_pos,input_h_pos)
#t_pos=tf.matmul(mrt_pos,input_t_pos)

score_hrt_pos1=tf.norm(input_h_pos+input_r_pos-input_t_pos,ord=1,axis=1)
#L1

score_hrt_pos121=tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)
score_hrt_pos12n=0.7*tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
0.3*tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)
score_hrt_posn21=0.3*tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
0.7*tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)
score_hrt_posn2n=tf.reduce_sum((input_h_pos+input_r_pos)*input_t_pos,axis=1)+\
tf.reduce_sum(input_h_pos*(input_t_pos-input_r_pos),axis=1)
#weight_*(tf.norm(input_h_pos,ord=2)+tf.norm(input_t_pos,ord=2)+tf.norm(input_r_pos,ord=2))

train_input_neg=tf.placeholder(tf.int32,[None,3])
#(nbatch,1)
input_h_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,0]),[-1,embed_dim,1])#(n_batch,1) (n_batch,dim)
hnn=tf.reshape(tf.norm(input_h_neg,axis=1,ord=2),[-1])
#hp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,0]),[n_batch,embed_dim,-1])
input_t_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,1]),[-1,embed_dim,1])
tnn=tf.reshape(tf.norm(input_t_neg,axis=1,ord=2),[-1])
#tp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,1]),[n_batch,embed_dim,-1])
input_r_neg=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_neg[:,2]),[-1,embed_dim,1])
rnn=tf.reshape(tf.norm(input_r_neg,axis=1,ord=2),[-1])
#rp_neg=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_neg[:,2]),[n_batch,embed_dim,-1])
diag_neg=tf.abs(tf.reshape(tf.nn.embedding_lookup(diag_matrix,train_input_neg[:,2]),[-1,embed_dim,1]))
#mrh_neg=tf.matmul(rp_neg,hp_neg,transpose_b=True)+tf.eye(embed_dim)
#mrt_neg=tf.matmul(rp_neg,tp_neg,transpose_b=True)+tf.eye(embed_dim)
#h_neg=tf.matmul(mrh_neg,input_h_neg)
#t_neg=tf.matmul(mrt_neg,input_t_neg)

eZeroNorm=tf.norm(tf.nn.embedding_lookup(ent_embedding,0))
rZeroNorm=tf.norm(tf.nn.embedding_lookup(rel_embedding,0))

score_hrt_neg1=tf.norm(input_h_neg+input_r_neg-input_t_neg,ord=1,axis=1)
#L1
score_hrt_neg121=tf.reduce_sum((input_h_neg+input_r_neg)*input_t_neg,axis=1)+\
tf.reduce_sum(input_h_neg*(input_t_neg-input_r_neg),axis=1)
score_hrt_neg12n=0.7*tf.reduce_sum((input_h_neg+input_r_neg)*input_t_neg,axis=1)+\
0.3*tf.reduce_sum(input_h_neg*(input_t_neg-input_r_neg),axis=1)

score_hrt_negn21=0.3*tf.reduce_sum((input_h_neg+input_r_neg)*input_t_neg,axis=1)+\
0.7*tf.reduce_sum(input_h_neg*(input_t_neg-input_r_neg),axis=1)
score_hrt_negn2n=tf.reduce_sum((input_h_neg+input_r_neg)*input_t_neg,axis=1)+\
tf.reduce_sum(input_h_neg*(input_t_neg-input_r_neg),axis=1)
#weight_*(tf.norm(input_h_neg,ord=2)+tf.norm(input_t_neg,ord=2)+tf.norm(input_r_neg,ord=2))


loss121=tf.reduce_sum(tf.nn.relu(margin_-score_hrt_pos121+score_hrt_neg121))+weight_*\
(tf.reduce_sum(tf.square(input_h_pos))+tf.reduce_sum(tf.square(input_r_pos))+\
	tf.reduce_sum(tf.square(input_t_pos))+tf.reduce_sum(tf.square(input_h_neg))+\
	tf.reduce_sum(tf.square(input_r_neg))+tf.reduce_sum(tf.square(input_t_neg)))

#weight_diag_*(tf.reduce_sum(tf.square(diag_pos))+tf.reduce_sum(tf.square(diag_neg)))
#+regularizer_weight*regularizer_loss
optimizer121=tf.train.GradientDescentOptimizer(learning_rate=lr)
#optimizer=tf.train.AdadeltaOptimizer()
grads121=optimizer121.compute_gradients(loss121,trainable)
op_train121=optimizer121.apply_gradients(grads121)
idx_e=tf.placeholder(tf.int32,[None])
idx_r=tf.placeholder(tf.int32,[None])
normedE=tf.nn.l2_normalize(tf.nn.embedding_lookup(ent_embedding,idx_e),axis=1)
normedR=tf.nn.l2_normalize(tf.nn.embedding_lookup(rel_embedding,idx_r),axis=1)
updateE=tf.scatter_update(ent_embedding,idx_e,normedE)
updateR=tf.scatter_update(rel_embedding,idx_r,normedR)

loss12n=tf.reduce_sum(tf.nn.relu(margin_-score_hrt_pos12n+score_hrt_neg12n))+weight_*\
(tf.reduce_sum(tf.square(input_h_pos))+tf.reduce_sum(tf.square(input_r_pos))+\
	tf.reduce_sum(tf.square(input_t_pos))+tf.reduce_sum(tf.square(input_h_neg))+\
	tf.reduce_sum(tf.square(input_r_neg))+tf.reduce_sum(tf.square(input_t_neg)))

optimizer12n=tf.train.GradientDescentOptimizer(learning_rate=lr)
grads12n=optimizer12n.compute_gradients(loss12n,trainable)
op_train12n=optimizer12n.apply_gradients(grads12n)

lossn21=tf.reduce_sum(tf.nn.relu(margin_-score_hrt_posn21+score_hrt_negn21))+weight_*\
(tf.reduce_sum(tf.square(input_h_pos))+tf.reduce_sum(tf.square(input_r_pos))+\
	tf.reduce_sum(tf.square(input_t_pos))+tf.reduce_sum(tf.square(input_h_neg))+\
	tf.reduce_sum(tf.square(input_r_neg))+tf.reduce_sum(tf.square(input_t_neg)))

optimizern21=tf.train.GradientDescentOptimizer(learning_rate=lr)
gradsn21=optimizern21.compute_gradients(lossn21,trainable)
op_trainn21=optimizern21.apply_gradients(gradsn21)

lossn2n=tf.reduce_sum(tf.nn.relu(margin_-score_hrt_posn2n+score_hrt_negn2n))+weight_*\
(tf.reduce_sum(tf.square(input_h_pos))+tf.reduce_sum(tf.square(input_r_pos))+\
	tf.reduce_sum(tf.square(input_t_pos))+tf.reduce_sum(tf.square(input_h_neg))+\
	tf.reduce_sum(tf.square(input_r_neg))+tf.reduce_sum(tf.square(input_t_neg)))

optimizern2n=tf.train.GradientDescentOptimizer(learning_rate=lr)
gradsn2n=optimizern2n.compute_gradients(lossn2n,trainable)
op_trainn2n=optimizern2n.apply_gradients(gradsn2n)

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
		ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			print("success! %s."% ckpt.model_checkpoint_path)
			saver.restore(sess,ckpt.model_checkpoint_path)
		else:
			print('fail to restore')
		n_epoch=0
		n_iter=0
		total=math.ceil(n_triple/n_batch)*num_epoch*2
		while n_epoch<num_epoch*2:
			n_epoch+=1
			if n_epoch>num_epoch:
				lr=lr2
			n_idx=0
			while(n_idx<n_triple):
				n_iter+=1
				if n_idx+n_batch>n_triple:
					input_pos=np.concatenate([train_triple[n_idx:n_idx+n_batch],train_triple[0:n_idx+n_batch-n_triple]],axis=0)
					train_triple=train_triple.tolist()
					random.shuffle(train_triple)
					train_triple=np.asarray(train_triple,dtype=np.int32)
					print("shuffled")
				else:
					input_pos=train_triple[n_idx:n_idx+n_batch]
				#input_pos=np.asarray(input_pos,dtype=np.int32)
				n_idx+=n_batch
				temp=input_pos.tolist()
				input_pos1=[]
				input_neg1=[]
				input_pos2=[]
				input_neg2=[]
				input_pos3=[]
				input_neg3=[]
				input_pos4=[]
				input_neg4=[]
				for idx in range(input_pos.shape[0]):
					if bern[input_pos[idx][2]][0]<=1.5 and bern[input_pos[idx][2]][1]<=1.5:
						input_pos1.append(input_pos[idx])
					elif bern[input_pos[idx][2]][0]>1.5 and bern[input_pos[idx][2]][1]<=1.5:
						input_pos2.append(input_pos[idx])
					elif bern[input_pos[idx][2]][0]<=1.5 and bern[input_pos[idx][2]][1]>1.5:
						input_pos3.append(input_pos[idx])
					elif bern[input_pos[idx][2]][0]>1.5 and bern[input_pos[idx][2]][1]>1.5:
						input_pos4.append(input_pos[idx])
					else:
						print('error')
				input_pos1=np.asarray(input_pos1,dtype=np.int32)
				input_pos2=np.asarray(input_pos2,dtype=np.int32)
				input_pos3=np.asarray(input_pos3,dtype=np.int32)
				input_pos4=np.asarray(input_pos4,dtype=np.int32)

				for idx in range(input_pos1.shape[0]):
					#if np.random.uniform(-1,1) > 0:
					if np.random.uniform(0,1) > bern[input_pos1[idx][2]][0]/(bern[input_pos1[idx][2]][0]+bern[input_pos1[idx][2]][1]):
						temp_ent=random.sample(entityID_list,1)[0]			
						input_neg1.append([int(temp_ent),temp[idx][1],temp[idx][2]])
						'''temp_ent=random.sample(Ehr[input_pos[idx][2]],1)[0]
						input_neg.append([temp_ent,temp[idx][1],temp[idx][2]])'''
					else:
						temp_ent=random.sample(entityID_list,1)[0]	
						input_neg1.append([temp[idx][0],int(temp_ent),temp[idx][2]])
						'''temp_ent=random.sample(Etr[input_pos[idx][2]],1)[0]
						input_neg.append([temp[idx][0],temp_ent,temp[idx][2]])'''
				for idx in range(input_pos2.shape[0]):
					#if np.random.uniform(-1,1) > 0:
					if np.random.uniform(0,1) > bern[input_pos2[idx][2]][0]/(bern[input_pos2[idx][2]][0]+bern[input_pos2[idx][2]][1]):
						temp_ent=random.sample(entityID_list,1)[0]			
						input_neg2.append([int(temp_ent),temp[idx][1],temp[idx][2]])
						'''temp_ent=random.sample(Ehr[input_pos[idx][2]],1)[0]
						input_neg.append([temp_ent,temp[idx][1],temp[idx][2]])'''
					else:
						temp_ent=random.sample(entityID_list,1)[0]	
						input_neg2.append([temp[idx][0],int(temp_ent),temp[idx][2]])
						'''temp_ent=random.sample(Etr[input_pos[idx][2]],1)[0]
						input_neg.append([temp[idx][0],temp_ent,temp[idx][2]])'''
				for idx in range(input_pos3.shape[0]):
					#if np.random.uniform(-1,1) > 0:
					if np.random.uniform(0,1) > bern[input_pos3[idx][2]][0]/(bern[input_pos3[idx][2]][0]+bern[input_pos3[idx][2]][1]):
						temp_ent=random.sample(entityID_list,1)[0]			
						input_neg3.append([int(temp_ent),temp[idx][1],temp[idx][2]])
						'''temp_ent=random.sample(Ehr[input_pos[idx][2]],1)[0]
						input_neg.append([temp_ent,temp[idx][1],temp[idx][2]])'''
					else:
						temp_ent=random.sample(entityID_list,1)[0]	
						input_neg3.append([temp[idx][0],int(temp_ent),temp[idx][2]])
						'''temp_ent=random.sample(Etr[input_pos[idx][2]],1)[0]
						input_neg.append([temp[idx][0],temp_ent,temp[idx][2]])'''
				for idx in range(input_pos4.shape[0]):
					#if np.random.uniform(-1,1) > 0:
					if np.random.uniform(0,1) > bern[input_pos4[idx][2]][0]/(bern[input_pos4[idx][2]][0]+bern[input_pos4[idx][2]][1]):
						temp_ent=random.sample(entityID_list,1)[0]			
						input_neg4.append([int(temp_ent),temp[idx][1],temp[idx][2]])
						'''temp_ent=random.sample(Ehr[input_pos[idx][2]],1)[0]
						input_neg.append([temp_ent,temp[idx][1],temp[idx][2]])'''
					else:
						temp_ent=random.sample(entityID_list,1)[0]	
						input_neg4.append([temp[idx][0],int(temp_ent),temp[idx][2]])
						'''temp_ent=random.sample(Etr[input_pos[idx][2]],1)[0]
						input_neg.append([temp[idx][0],temp_ent,temp[idx][2]])'''

				input_neg1=np.asarray(input_neg1,dtype=np.int32)
				input_neg2=np.asarray(input_neg2,dtype=np.int32)
				input_neg3=np.asarray(input_neg3,dtype=np.int32)
				input_neg4=np.asarray(input_neg4,dtype=np.int32)
				#print(input_neg)
				if input_pos1.shape[0]!=0:
					hp,loss_iter,_=sess.run([hpn,loss121,op_train121],{train_input_pos:input_pos1,train_input_neg:input_neg1})
					loss_sum+=loss_iter
				if input_pos2.shape[0]!=0:
					hp,loss_iter,_=sess.run([hpn,loss12n,op_train12n],{train_input_pos:input_pos2,train_input_neg:input_neg2})
					loss_sum+=loss_iter
				if input_pos3.shape[0]!=0:
					hp,loss_iter,_=sess.run([hpn,lossn21,op_trainn21],{train_input_pos:input_pos3,train_input_neg:input_neg3})
					loss_sum+=loss_iter
				if input_pos4.shape[0]!=0:
					hp,loss_iter,_=sess.run([hpn,lossn2n,op_trainn2n],{train_input_pos:input_pos4,train_input_neg:input_neg4})
					loss_sum+=loss_iter


				'''hp=(hp>1)*input_pos[:,0]
				tp=(tp>1)*input_pos[:,1]
				rp=(rp>1)*input_pos[:,2]
				hn=(hn>1)*input_neg[:,0]
				tn=(tn>1)*input_neg[:,1]
				rn=(rn>1)*input_neg[:,2]# id of the vectors which need to be normalized

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
				sess.run([updateE,updateR],{idx_e:norm_elist,idx_r:norm_rlist})'''
				if n_iter%100==0:

					print(hp[:10])
					print(n_iter,'/',total,' learning rate:',lr)
					print(loss_sum)
					loss_sum=0
					saved_path=saver.save(sess,checkpoint_dir+model_name)
		print("complete")
		saved_path=saver.save(sess,checkpoint_dir+model_name)
		print("saved\n",saved_path)
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
			while(n_idx<n_triple):
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
				if bern[input_pos[0][2]][0]<=1.5 and bern[input_pos[0][2]][1]<=1.5:
					scores=sess.run(score_hrt_pos121,{train_input_pos:input_list})
				elif bern[input_pos[0][2]][0]>1.5 and bern[input_pos[0][2]][1]<=1.5:
					scores=sess.run(score_hrt_pos12n,{train_input_pos:input_list})
				elif bern[input_pos[0][2]][0]<=1.5 and bern[input_pos[0][2]][1]>1.5:
					scores=sess.run(score_hrt_posn21,{train_input_pos:input_list})
				elif bern[input_pos[0][2]][0]>1.5 and bern[input_pos[0][2]][1]>1.5:
					scores=sess.run(score_hrt_posn2n,{train_input_pos:input_list})
				else:
					print("error_test")
				scores=scores.reshape(-1).tolist()
				#index=temp_idx
				temp=scores[index]
				#temp=scores[temp_idx]
				scores.sort(reverse=True)
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
				if bern[input_pos[0][2]][0]<=1.5 and bern[input_pos[0][2]][1]<=1.5:
					scores=sess.run(score_hrt_pos121,{train_input_pos:input_list})
				elif bern[input_pos[0][2]][0]>1.5 and bern[input_pos[0][2]][1]<=1.5:
					scores=sess.run(score_hrt_pos12n,{train_input_pos:input_list})
				elif bern[input_pos[0][2]][0]<=1.5 and bern[input_pos[0][2]][1]>1.5:
					scores=sess.run(score_hrt_posn21,{train_input_pos:input_list})
				elif bern[input_pos[0][2]][0]>1.5 and bern[input_pos[0][2]][1]>1.5:
					scores=sess.run(score_hrt_posn2n,{train_input_pos:input_list})
				else:
					print("error_test")
				#scores=sess.run(score_hrt_pos,{train_input_pos:input_list})
				scores=scores.reshape(-1).tolist()
				#index=temp_idx
				#temp=scores[temp_idx]
				temp=scores[index]
				scores.sort(reverse=True)
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