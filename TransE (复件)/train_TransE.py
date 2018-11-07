import tensorflow as tf
import numpy as np
import csv
import math
import random
embed_dim=20
n_batch=512
margin=2.
lr=0.0001
regularizer_weight=0.00001
num_epoch=1
#train_path='/data/Relation_Extraction/data/WN18/train.txt'
train_path='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/train.txt'
#checkpoint_dir='/data/Relation_Extraction/data/WN18/saver/'
checkpoint_dir='/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/saver/'
model_name='modeld'
entity_id_map={}
id_entity_map={}
relation_id_map={}
id_relation_map={}
#csv_file=csv.reader(open('/data/Relation_Extraction/data/WN18/entity2id.txt'))
csv_file=csv.reader(open('/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/entity2id.txt'))
n_entity=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_entity+=1
	entity_id_map[line[0]]=line[1]
	#id_entity_map[line[1]]=line[0]

#csv_file=csv.reader(open('/data/Relation_Extraction/data/WN18/relation2id.txt'))
csv_file=csv.reader(open('/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/relation2id.txt'))
n_relation=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_relation+=1
	relation_id_map[line[0]]=line[1]
	#id_relation_map[line[1]]=line[0]
print("entity number:%d,relation number:%d"%(n_entity,n_relation))
#print(entity_id_map)
entity_list=list(entity_id_map.values())
relation_list=list(relation_id_map.values())
def load_triple(file_path):
	with open(file_path,'r',encoding='utf-8') as f_triple:
		return np.asarray([[entity_id_map[x.strip().split('\t')[0]],
			entity_id_map[x.strip().split('\t')[1]],
			relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
			dtype=np.int32)
train_triple=load_triple(train_path)

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

#hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])

input_t_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,1]),[n_batch,embed_dim,-1])

#tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])

input_r_pos=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_pos[:,2]),[n_batch,embed_dim,-1])

#rp_pos=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_pos[:,2]),[n_batch,embed_dim,-1])

#mrh_pos=tf.matmul(rp_pos,hp_pos,transpose_b=True)+tf.eye(embed_dim)
#mrt_pos=tf.matmul(rp_pos,tp_pos,transpose_b=True)+tf.eye(embed_dim)
#h_pos=tf.matmul(mrh_pos,input_h_pos)
#t_pos=tf.matmul(mrt_pos,input_t_pos)
score_hrt_pos=tf.norm(input_h_pos+input_r_pos-input_t_pos,ord=1,axis=1)
train_input_neg=tf.placeholder(tf.int32,[None,3])
#(nbatch,1)
input_h_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,0]),[n_batch,embed_dim,-1])#(n_batch,1) (n_batch,dim)
#hp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,0]),[n_batch,embed_dim,-1])
input_t_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,1]),[n_batch,embed_dim,-1])
#tp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,1]),[n_batch,embed_dim,-1])
input_r_neg=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_neg[:,2]),[n_batch,embed_dim,-1])
#rp_neg=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_neg[:,2]),[n_batch,embed_dim,-1])
#mrh_neg=tf.matmul(rp_neg,hp_neg,transpose_b=True)+tf.eye(embed_dim)
#mrt_neg=tf.matmul(rp_neg,tp_neg,transpose_b=True)+tf.eye(embed_dim)
#h_neg=tf.matmul(mrh_neg,input_h_neg)
#t_neg=tf.matmul(mrt_neg,input_t_neg)

score_hrt_neg=tf.norm(input_h_neg+input_r_neg-input_t_neg,ord=1,axis=1)

regularizer_loss=tf.abs(tf.reduce_sum(ent_embedding)+tf.reduce_sum(rel_embedding))

loss=tf.reduce_sum(tf.nn.relu(score_hrt_pos-score_hrt_neg+margin_))+regularizer_weight*regularizer_loss

optimizer=tf.train.GradientDescentOptimizer(learning_rate=lr)
grads=optimizer.compute_gradients(loss,trainable)
op_train=optimizer.apply_gradients(grads)

saver=tf.train.Saver()




'''B=np.array([[1.,2.,3.,4.],[2.,2.,3.,4.]])
A=tf.Variable(B)
C=tf.nn.embedding_lookup(A,[1])
D=tf.nn.l2_normalize(C)

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1)
grads=optimizer.compute_gradients(D)
op_train=optimizer.apply_gradients(grads)'''


#filenames=['../data/WN18/entity2id.txt','../data/WN18/relation2id.txt']
#filenames=['/data/Relation_Extraction/data/WN18/train.txt']
filenames=['/media/ggxxding/documents/GitHub/ggxxding/Relation_Extraction/data/WN18/test.txt']
filename_queue=tf.train.string_input_producer(filenames,shuffle=False,num_epochs=num_epoch)
#num_epochs 迭代轮数，每个数据最多出现多少次
reader=tf.TextLineReader()
key,value=reader.read(filename_queue)
record_defaults=[['NULL'],['NULL'],['NULL']]
col1,col2,col3=tf.decode_csv(value,record_defaults=record_defaults,field_delim="\t")
#features=tf.stack([col1,col2])
col1_batch,col2_batch,col3_batch=tf.train.batch([col1,col2,col3],batch_size=n_batch)


# 启动图 (graph)
init=tf.global_variables_initializer()
init_local_op=tf.initialize_local_variables()
with tf.Session() as sess:
	n_iter=0
	sess.run(init)
	sess.run(init_local_op)


	ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		print("success! %s."% ckpt.model_checkpoint_path)
		saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		print('fail to restore')

	coord=tf.train.Coordinator()
	threads=tf.train.start_queue_runners(coord=coord)
	try:
		while not coord.should_stop():
			n_iter+=1

			c1,c2,c3=sess.run([col1_batch,col2_batch,col3_batch])
			input_pos=[]
			for idx in range(len(c1)):
				input_pos.append([entity_id_map[bytes.decode(c1[idx])],entity_id_map[bytes.decode(c2[idx])], \
					relation_id_map[bytes.decode(c3[idx])]])
			input_pos=np.asarray(input_pos,dtype=np.int32)
			#print(input_pos,input_pos.shape[0])

			temp=input_pos.tolist()
			print(input_pos[0:2])
			input_neg=[]
			for idx in range(input_pos.shape[0]):
				if np.random.uniform(-1,1) > 0:
					'''flag1=0
					flag2=0
					while flag1==0 or flag2==0 :
						flag1=0
						flag2=0
						temp_ent=random.sample(list(entity_id_map.values()),1)[0]
						if int(temp_ent) != temp[idx][0]:
							flag1=1
						if flag1==1:
							for idx1 in range(train_triple.shape[0]):
								if ([int(temp_ent),temp[idx][1],temp[idx][2]]== \
									[train_triple[idx1][0],train_triple[idx1][1],train_triple[idx1][2]]):
									break
								if idx1==(train_triple.shape[0]-1):
									flag2=1'''
					temp_ent=random.sample(entity_list,1)[0]			
					input_neg.append([int(temp_ent),temp[idx][1],temp[idx][2]])
				else:
					'''flag1=0
					flag2=0
					while flag1==0 or flag2==0:
						flag1=0
						flag2=0
						temp_ent=random.sample(list(entity_id_map.values()),1)[0]
						if int(temp_ent) != temp[idx][2]:
							flag1=1
						if flag1==1:
							for idx1 in range(train_triple.shape[0]):
								if ([temp[idx][0],int(temp_ent),temp[idx][2]]== \
									[train_triple[idx1][0],train_triple[idx1][1],train_triple[idx1][2]]):
									break
								if idx1==(train_triple.shape[0]-1):
									flag2=1'''
					temp_ent=random.sample(entity_list,1)[0]	
					input_neg.append([temp[idx][0],int(temp_ent),temp[idx][2]])
			input_neg=np.array(input_neg,dtype=np.int32)
			#print(input_neg)
			losss,_=sess.run([loss,op_train],{train_input_pos:input_pos,train_input_neg:input_neg})
			print(losss)

			if n_iter%100==0:
				print(n_iter*n_batch,'/',num_epoch*len(train_triple))
				saved_path=saver.save(sess,checkpoint_dir+model_name)

	except tf.errors.OutOfRangeError:
		print('epochs complete!')
	finally:
		coord.request_stop()
	coord.join(threads)
	coord.request_stop()
	coord.join(threads)
	saver.save(sess,checkpoint_dir+model_name)


# 拟合平面
'''for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, W.eval(sess), sess.run(b))'''

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]