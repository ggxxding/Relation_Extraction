import tensorflow as tf
import numpy as np
import csv
dim=10
entity_Dict={}
relation_Dict={}
csv_file=csv.reader(open('../data/WN182/entity2id.txt'))
entity_num=0
for lines in csv_file:
	line=lines[0].split('\t')
	entity_num+=1
	entity_Dict[line[0]]=int(line[1])

csv_file=csv.reader(open('../data/WN182/relation2id.txt'))
relation_num=0
for lines in csv_file:
	line=lines[0].split('\t')
	relation_num+=1
	relation_Dict[line[0]]=int(line[1])
print("entity number:%d,relation number:%d"%(entity_num,relation_num))

#tf.placeholder()

e=[]
e_p=[]
for i in range(entity_num):
	e.append(tf.Variable(tf.random_uniform([dim,1],-1.0,1.0)))
	e_p.append(tf.Variable(tf.random_uniform([dim,1],-1.0,1.0)))
	if i%1000==0:
		print('initializing entity:%d/%d'%(i,entity_num))

r=[]
r_p=[]
for i in range(relation_num):
	r.append(tf.Variable(tf.random_uniform([dim,1],-1.0,1.0)))
	r_p.append(tf.Variable(tf.random_uniform([dim,1],-1.0,1.0)))
	print('initializing relation:%d/%d'%(i,relation_num))


#filenames=['../data/WN18/entity2id.txt','../data/WN18/relation2id.txt']
filenames=['../data/WN182/test.txt']
filename_queue=tf.train.string_input_producer(filenames,shuffle=False,num_epochs=1)
#num_epochs 迭代轮数，每个数据最多出现多少次
reader=tf.TextLineReader()
key,value=reader.read(filename_queue)
record_defaults=[['NULL'],['NULL'],['NULL']]
col1,col2,col3=tf.decode_csv(value,record_defaults=record_defaults,field_delim="\t")
#features=tf.stack([col1,col2])
col1_batch,col2_batch,col3_batch=tf.train.shuffle_batch([col1,col2,col3],batch_size=1,capacity=200,min_after_dequeue=100)

#0821
'''
head=tf.placeholder(tf.int32, [None, 1])
tail=tf.placeholder(tf.int32,[None,1])
rel=tf.placeholder(tf.int32,[None,1])'''
'''
m_rh=tf.matmul(r_p[rel],tf.transpose(e_p[head]))+tf.eye(dim)
h_=tf.matmul(m_rh,e[head])
m_rt=tf.matmul(r_p[rel],tf.transpose(e_p[tail]))+tf.eye(dim)
t_=tf.matmul(m_rt,e[tail])'''

# 启动图 (graph)
init=tf.global_variables_initializer()
init_local_op=tf.initialize_local_variables()
with tf.Session() as sess:

	sess.run(init)
	sess.run(init_local_op)
	'''
	print(h_p.eval(),r_p.eval())
	print(tf.matmul(h_p,tf.transpose(r_p)).eval())'''

	coord=tf.train.Coordinator()
	threads=tf.train.start_queue_runners(coord=coord)
	'''for i in range(10):
		c1,c2=sess.run([col1_batch,col2_batch])
		print(c1,c2)'''
	try:
		while not coord.should_stop():
			c1,c2,c3=sess.run([col1_batch,col2_batch,col3_batch])
			c1=bytes.decode(c1[0])
			c2=bytes.decode(c2[0])
			c3=bytes.decode(c3[0])
			print(c1,c2,c3)

	except tf.errors.OutOfRangeError:
		print('epochs complete!')
	finally:
		coord.request_stop()
	coord.join(threads)
	coord.request_stop()
	coord.join(threads)

with tf.Session() as sess:
	sess.run(init)
# 拟合平面
'''for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, W.eval(sess), sess.run(b))'''

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]