import tensorflow as tf
import numpy as np
import csv
import math
import random

embed_dim=20
n_batch=1#
num_epoch=1

test_path='../data/wn182/ttt.txt'
checkpoint_dir='../data/wn182/saver/'
model_name='modeld'
entity_id_map={}
id_entity_map={}
relation_id_map={}
id_relation_map={}
csv_file=csv.reader(open('../data/wn182/entity2id.txt'))
n_entity=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_entity+=1
	entity_id_map[line[0]]=line[1]
	#id_entity_map[line[1]]=line[0]

csv_file=csv.reader(open('../data/wn182/relation2id.txt'))
n_relation=0
for lines in csv_file:
	line=lines[0].split('\t')
	n_relation+=1
	relation_id_map[line[0]]=line[1]
	#id_relation_map[line[1]]=line[0]
print("entity number:%d,relation number:%d"%(n_entity,n_relation))
#print(entity_id_map)

def load_triple(file_path):
	with open(file_path,'r',encoding='utf-8') as f_triple:
		return np.asarray([[entity_id_map[x.strip().split('\t')[0]],
			entity_id_map[x.strip().split('\t')[1]],
			relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
			dtype=np.int32)
test_triple=load_triple(test_path)

#tf.placeholder()
trainable=[] #可训练参数列表


bound = 6 / math.sqrt(embed_dim)
ent_embedding = tf.get_variable("ent_embedding", [n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=345))
ent_projecting=tf.get_variable("ent_projecting", [n_entity, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=347))
trainable.append(ent_embedding)
trainable.append(ent_projecting)

rel_embedding = tf.get_variable("rel_embedding", [n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=346))
rel_projecting=tf.get_variable("rel_projecting", [n_relation, embed_dim],
                                                   initializer=tf.random_uniform_initializer(minval=-bound, \
                                                   	maxval=bound,seed=348))
trainable.append(rel_embedding)
trainable.append(rel_projecting)


train_input_pos=tf.placeholder(tf.int32,[None,3])#(nbatch,1)
input_h_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,0]),[n_batch,embed_dim,-1])#(n_batch,1) (n_batch,dim)
hp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,0]),[n_batch,embed_dim,-1])
input_t_pos=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_pos[:,1]),[n_batch,embed_dim,-1])
tp_pos=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_pos[:,1]),[n_batch,embed_dim,-1])
input_r_pos=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_pos[:,2]),[n_batch,embed_dim,-1])
rp_pos=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_pos[:,2]),[n_batch,embed_dim,-1])
mrh_pos=tf.matmul(rp_pos,hp_pos,transpose_b=True)+tf.eye(embed_dim)
mrt_pos=tf.matmul(rp_pos,tp_pos,transpose_b=True)+tf.eye(embed_dim)
h_pos=tf.matmul(mrh_pos,input_h_pos)
t_pos=tf.matmul(mrt_pos,input_t_pos)

score_hrt_pos=tf.square(tf.norm(h_pos+input_r_pos-t_pos,axis=1))


'''train_input_neg=tf.placeholder(tf.int32,[None,3])#(nbatch,1)
input_h_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,0]),[n_batch,embed_dim,-1])#(n_batch,1) (n_batch,dim)
hp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,0]),[n_batch,embed_dim,-1])
input_t_neg=tf.reshape(tf.nn.embedding_lookup(ent_embedding,train_input_neg[:,1]),[n_batch,embed_dim,-1])
tp_neg=tf.reshape(tf.nn.embedding_lookup(ent_projecting,train_input_neg[:,1]),[n_batch,embed_dim,-1])
input_r_neg=tf.reshape(tf.nn.embedding_lookup(rel_embedding,train_input_neg[:,2]),[n_batch,embed_dim,-1])
rp_neg=tf.reshape(tf.nn.embedding_lookup(rel_projecting,train_input_neg[:,2]),[n_batch,embed_dim,-1])
mrh_neg=tf.matmul(rp_neg,hp_neg,transpose_b=True)+tf.eye(embed_dim)
mrt_neg=tf.matmul(rp_neg,tp_neg,transpose_b=True)+tf.eye(embed_dim)
h_neg=tf.matmul(mrh_neg,input_h_neg)
t_neg=tf.matmul(mrt_neg,input_t_neg)
score_hrt_neg=tf.square(tf.norm(h_neg+input_r_neg-t_neg,axis=1))'''
'''regularizer_loss=tf.reduce_sum(ent_embedding)+tf.reduce_sum(ent_projecting) \
+tf.reduce_sum(rel_embedding)+tf.reduce_sum(rel_projecting)'''


saver=tf.train.Saver()


#filenames=['../data/WN18/entity2id.txt','../data/WN18/relation2id.txt']
filenames=['../data/wn182/ttt.txt']
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
test_triple=test_triple.tolist()
with tf.Session() as sess:
	n_iter=0
	n_loading=0
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
	rank_list=[]
	try:
		while not coord.should_stop():
			n_iter+=1
			c1,c2,c3=sess.run([col1_batch,col2_batch,col3_batch])
			input_pos=[]
			for idx in range(len(c1)):
				input_pos.append([entity_id_map[bytes.decode(c1[idx])],entity_id_map[bytes.decode(c2[idx])], \
					relation_id_map[bytes.decode(c3[idx])]])
			input_pos=np.asarray(input_pos,dtype=np.int32)
			score_list=[]
			#print(input_pos,input_pos.shape[0])

			scores=sess.run(score_hrt_pos,{train_input_pos:input_pos})

			score_list.append(scores[0][0])
			temp=input_pos.tolist()#temp=[[]]

			for idx in range(len(entity_id_map)):
				if (idx!=temp[0][0]) and ([idx,temp[0][1],temp[0][2]] not in test_triple):
					scores_corrupted=sess.run(score_hrt_pos,{train_input_pos:np.asarray([[idx,temp[0][1],temp[0][2]]],dtype=np.int32)})
					score_list.append(scores_corrupted[0][0])

			#corrupted_input=np.array(input_neg,dtype=np.int32)




			n_loading+=1
			score_list.sort()
			rank=score_list.index(scores[0][0])
			rank_list.append(rank)

			print(n_loading)
			print(score_list.index(scores[0][0]))


			'''if n_iter%100==0:
				saved_path=saver.save(sess,checkpoint_dir+model_name)'''

	except tf.errors.OutOfRangeError:
		print('epochs complete!')
	finally:
		coord.request_stop()
	coord.join(threads)
	coord.request_stop()
	coord.join(threads)
	hits10=0
	for i in rank_list:
		if i<=10:
			hits10+=1
	hits10=hits10/len(rank_list)
	rank_list=np.asarray(rank_list,dtype=np.int32)
	mean_rank=np.sum(rank_list)/rank_list.shape[0]
	print('hits10:',hits10)
	print('meanrank:',mean_rank)


	#saver.save(sess,checkpoint_dir+model_name)


# 拟合平面
'''for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, W.eval(sess), sess.run(b))'''

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]