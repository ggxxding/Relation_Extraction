import tensorflow as tf
import numpy as np
# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300
#tf.placeholder()
# 构造一个线性模型
dim=10
h = tf.Variable(tf.random_uniform([dim,1],-1.0,1.0))
h_p = tf.Variable(tf.random_uniform([dim,1],-1.0,1.0))
r=tf.Variable(tf.random_uniform([dim,1],-1.0,1.0))
r_p = tf.Variable(tf.random_uniform([dim,1],-1.0,1.0))

# 最小化方差
#loss = tf.reduce_mean(tf.square(y - y_data))
#optimizer = tf.train.GradientDescentOptimizer(0.5)
#train = optimizer.minimize(loss)

# 启动图 (graph)


with tf.Session() as sess:
	init=tf.global_variables_initializer()
	sess.run(init)
	print(h_p.eval(),r_p.eval())
	print(tf.matmul(h_p,tf.transpose(r_p)).eval())
# 拟合平面
'''for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, W.eval(sess), sess.run(b))'''

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]