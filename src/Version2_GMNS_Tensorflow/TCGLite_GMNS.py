#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# datetime: 2020/7/7 9:41
# software: PyCharm
# BTCGLite_GMNS.py

# In[0] Setting
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None  # Do not show the copy warning
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100
import numpy as np
from keras import backend as K

NUM_PARALLEL_EXEC_UNITS = 8

inputLocation = "SiouxFalls network/"
outputLocation = "SiouxFalls network/"+"BTCGLite/"
picLocation = inputLocation + "pictures/"


config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})


session = tf.Session(config=config)

K.set_session(session)

import os

os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


# In[0] Functions
# The adjacent layers are connected and normalized
def connection(input_layer, trans_mat, inc_mat):
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tm = tf.multiply(trans_mat, inc_mat)
    trans_mat = tf.transpose(tf.transpose(tm) / tf.reduce_sum(tm, 1))
    layer = tf.matmul(input_layer, trans_mat)
    return (tf.nn.relu(layer))


# initial the variables
def init_variable(shape, initial_value, layer_name):
    estimation = tf.Variable(tf.random_normal(shape, mean=initial_value, stddev=0, dtype=tf.float64, seed=1),
                             name=layer_name)
    return estimation


def build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile, f_count, lr_count, loss, learning_rate):
    # def build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile,f_count, lr_count):alpha，gamma，count
    t0 = datetime.datetime.now()
    my_opt_survey = tf.train.GradientDescentOptimizer(lr_survey)
    opt_survey = my_opt_survey.minimize(f_survey)
    t1 = datetime.datetime.now()
    print('Building Optimizer survey using Time: ', t1 - t0, '\n')

    my_opt_mobile = tf.train.GradientDescentOptimizer(lr_mobile)
    opt_mobile = my_opt_mobile.minimize(f_mobile)
    t2 = datetime.datetime.now()
    print('Building Optimizer mobile using Time: ', t2 - t0, '\n')

    my_opt_count = tf.train.GradientDescentOptimizer(lr_count)
    opt_count = my_opt_count.minimize(f_count)
    t3 = datetime.datetime.now()
    print('Building Optimizer sensor using Time: ', t3 - t0, '\n')

    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    opt_total = my_opt.minimize(loss)
    return opt_survey, opt_mobile, opt_count, opt_total



if __name__ == '__main__':
    # In[1] Input all the samples
    print('----Step 1: Input data----', '\n')
    t0 = datetime.datetime.now()
    all_node_df = pd.read_csv(inputLocation + 'node.csv', encoding='gbk')
    all_link_df = pd.read_csv(inputLocation + 'road_link.csv', encoding='gbk')
    all_agent_df = pd.read_csv(inputLocation + 'agent.csv', encoding='gbk')
    agent_type_df=pd.read_csv(inputLocation + 'agent_type.csv', encoding='gbk')


    # In[2] Only consider the first sample
    ozone_df = all_agent_df[all_agent_df['agent_type'] == 1]
    ozone_df.rename(columns={'agent_id':'ozone_id','observations': 'trip_generation'}, inplace=True)
    od_df = all_agent_df[all_agent_df['agent_type'] == 2]
    od_df.rename(columns={'agent_id':'od_id','observations': 'OD_split'}, inplace=True)
    path_df = all_agent_df[all_agent_df['agent_type'] == 3]
    path_df.rename(columns={'agent_id':'path_id','observations': 'path_proportion'}, inplace=True)
    link_df = all_agent_df[all_agent_df['agent_type'] == 4]
    link_df.rename(columns={'agent_id':'link_id','observations': 'sensor_count'}, inplace=True)

    link_with_sensor_df = link_df[link_df['sensor_count'] >= 0]

    ozone_df.reset_index(drop=True, inplace=True)
    od_df.reset_index(drop=True, inplace=True)
    path_df.reset_index(drop=True, inplace=True)
    link_df.reset_index(drop=True, inplace=True)

    # In[3] Basic parameters
    num_survey = 1
    num_mobile = 1
    num_float = 1
    num_sensor = 1
    num_ozone = ozone_df.shape[0]
    num_od = od_df.shape[0]
    num_path = path_df.shape[0]
    num_link = link_df.shape[0]

    batch_size = 1

    t1 = datetime.datetime.now()
    print('Input data using time', t1 - t0, '\n')

    # In[4] Build up dicts

    print('----Step 2: Build up Hash tables----', '\n')
    t0 = datetime.datetime.now()
    print('Dictionary on link layer...')
    link_df['link_pair'] = link_df.apply(lambda x: (int(x.o_node_id), int(x.d_node_id)), axis=1)  # map each link records to the internal data
    link_id_pair_dict = link_df[['link_id', 'link_pair']].set_index('link_pair').to_dict()['link_id']

    print('Dictionary on ozone layer...')
    node_zone_dict = ozone_df[['o_node_id', 'ozone_id']].set_index('o_node_id').to_dict()['ozone_id']

    print('Dictionary on od layer...')
    od_df['od_pair'] = od_df.apply(lambda x: (int(x.o_zone_id), int(x.d_zone_id)), axis=1)
    od_pair_dict = od_df[['od_pair', 'od_id']].set_index('od_pair').to_dict()['od_id']
    od_df['ozone_id'] = od_df.apply(lambda x: node_zone_dict[int(x.o_zone_id)], axis=1)

    print('Dictionary from path to od...')
    path_df['od_id'] = path_df.apply(lambda x: od_pair_dict[int(x.o_zone_id), int(x.d_zone_id)], axis=1)
    path_od_dict = path_df[['path_id', 'od_id']].set_index('path_id').to_dict()['od_id']

    print('Dictonary from od to o...')
    od_ozone_dict = od_df[['ozone_id', 'od_id']].set_index('od_id').to_dict()['ozone_id']

    t1 = datetime.datetime.now()
    print('\n', 'CPU time:', t1 - t0, '\n')

    # In[5] Build up Data set
    print('----Step 3: Build up data sets----', '\n')
    print('Introduce samples...')
    t0 = datetime.datetime.now()
    survey_val = np.zeros(shape=[num_survey, num_ozone])
    for s in range(num_survey):
        survey_val[s] = ozone_df[ozone_df.time_peroid == s + 1].trip_generation

    sensor_val = np.zeros(shape=[num_sensor, num_link])
    for s in range(num_sensor):
        print(s,link_df)
        sensor_val[s] = link_df[link_df['time_peroid']== s+1].sensor_count
        print(sensor_val)

    mobile_val = np.zeros(shape=[num_mobile, num_ozone, num_od])
    for m in range(num_mobile):
        temp_matrix = np.zeros([num_ozone, num_od])
        for i in range(num_od):
            od_w = od_df.loc[i]
            print(od_w)
            temp_matrix[int(od_w.ozone_id - 1)][int(od_w.od_id-1)] = od_w.OD_split
        mobile_val[m] = temp_matrix

    print('Build up Placeholders...')
    survey_train = tf.placeholder(shape=[batch_size, num_ozone], dtype=tf.float64)

    mobile_train = tf.placeholder(shape=[batch_size, num_ozone, num_od], dtype=tf.float64)
    sensor_train = tf.placeholder(shape=[batch_size, num_link], dtype=tf.float64)
    t1 = datetime.datetime.now()
    print('\n', 'CPU time:', t1 - t0, '\n')

    # In[6] Build up incident matrices
    print('Step 4: Build up incident matrices', '\n')
    t0 = datetime.datetime.now()

    print('Link_id to path_id incident matrix...')
    path_link_inc_mat = np.zeros([num_path, num_link])
    for i in range(num_path):
        path_r = path_df.loc[i]
        node_list = path_r.node_sequence.split(';')
        for link_l in range(len(node_list) - 1):
            link_pair = (int(node_list[len(node_list) - 1 - link_l]), int(node_list[len(node_list) - 2 - link_l]))
            link_id = link_id_pair_dict[link_pair]
            path_link_inc_mat[int(path_r.path_id - 1)][int(link_id - 1)] = 1.0

    print('Path to od incident matrix...')
    od_path_inc_mat = np.zeros([num_od, num_path])
    for i in range(num_path):
        path_r = path_df.loc[i]
        path_id = path_r.path_id
        od_id = path_od_dict[path_id]
        od_path_inc_mat[int(od_id - 1)][int(path_id - 1)] = 1.0
    print(od_path_inc_mat)

    print('Od to ozone incident matrix...')
    ozone_od_inc_mat = np.zeros([num_ozone, num_od])
    for i in range(num_od):
        od_w = od_df.loc[i]
        od_id = od_w.od_id
        ozone_id = od_ozone_dict[od_id]
        ozone_od_inc_mat[int(ozone_id - 1)][int(od_id - 1)] = 1.0

    #Filter sensor >0,and normalization
    sensor_mat = np.array(link_df.sensor_count >= 0).astype(float)
    sensor_mat = sensor_mat / np.sum(sensor_mat)
    print(sensor_mat)
    t1 = datetime.datetime.now()
    print('\n', 'CPU time:', t1 - t0, '\n')

    # In[8] Build up Computational graph
    print('----Step 4: Build up Computational graph----', '\n')
    print('Create layer from ozone to od...')
    est_alpha = init_variable([1, num_ozone], 60, 'alpha_')
    est_alpha = tf.nn.relu(est_alpha)
    est_gamma = init_variable([num_ozone, num_od], 1, 'gamma_')
    est_gamma = tf.nn.relu(est_gamma)
    alpha_1=tf.constant(1, shape=[1, num_ozone],dtype=tf.float64)
    est_q = connection(est_alpha, est_gamma, ozone_od_inc_mat)
    est_gamma1=connection(alpha_1, est_gamma, ozone_od_inc_mat)
    print('Create layer from od to path...')
    est_rou = init_variable([num_od, num_path], 1, 'rou_')
    est_rou = tf.nn.relu(est_rou)
    q_1=tf.constant(1, shape=[1, num_od],dtype=tf.float64)
    est_f = connection(est_q, est_rou, od_path_inc_mat)  # q*rou
    est_rou1=connection(q_1, est_rou, od_path_inc_mat)
    print('Create layer from path to link...')
    est_v = tf.nn.relu(tf.matmul(est_f, path_link_inc_mat))

    # In[9] Build up loss functions
    print('----Step 5: Build up loss function----', '\n')
    t0 = datetime.datetime.now()
    f_survey = tf.reduce_mean(tf.pow(tf.subtract(est_alpha, survey_train), 2))

    f_mobile = tf.reduce_mean(tf.pow(tf.subtract(tf.multiply(est_gamma, ozone_od_inc_mat), mobile_train), 2))

    f_count = tf.reduce_mean(
        tf.pow(tf.subtract(tf.multiply(est_v, sensor_mat), tf.multiply(sensor_train, sensor_mat)), 2))

    loss = f_survey + f_mobile + f_count

    maximum_iterations = 1001
    curr_iter = tf.Variable(0)  # Current iteration times
    # learning rate exponential decay
    lr_survey = tf.train.exponential_decay(0.5, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)
    lr_mobile = tf.train.exponential_decay(0.99, curr_iter, decay_steps=maximum_iterations / 20, decay_rate=0.99)
    lr_count = tf.train.exponential_decay(0.3, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)
    learning_rate = tf.train.exponential_decay(1e-2, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)

    opt_survey, opt_mobile, opt_count, opt_total = build_optimizer(f_survey, lr_survey, f_mobile, lr_mobile, f_count,
                                                                   lr_count, loss, learning_rate)

    t1 = datetime.datetime.now()
    print('\n', 'CPU time:', t1 - t0, '\n')
    print('----Step 6: Start training----', '\n')
    tt0 = datetime.datetime.now()
    with tf.Session(config=config) as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        list_survey = []
        list_mobile = []
        list_sensor = []
        list_total = []
        for nb_iter in range(maximum_iterations):
            # (1) Select a sample randomly
            t0 = datetime.datetime.now()
            rand_survey_index = np.random.choice(num_survey, batch_size)
            rand_mobile_index = np.random.choice(num_mobile, batch_size)
            rand_sensor_index = np.random.choice(num_sensor, batch_size)
            feed = {}
            feed[survey_train] = survey_val[rand_survey_index]
            feed[mobile_train] = mobile_val[rand_mobile_index]
            feed[sensor_train] = sensor_val[rand_sensor_index]

            train_total, _ = sess.run([loss, opt_total], feed_dict=feed)
            train_survey, _ = sess.run([f_survey, opt_survey], feed_dict=feed)
            train_mobile, _ = sess.run([f_mobile, opt_mobile], feed_dict=feed)
            train_sensor, _ = sess.run([f_count, opt_count], feed_dict=feed)

            #train_total = (train_survey + train_mobile + train_sensor) / 3

            list_survey.append(train_survey)
            list_mobile.append(train_mobile)
            list_sensor.append(train_sensor)
            list_total.append(train_total)
            if nb_iter % 100 == 0:
                print('step', nb_iter, ':survey error=', train_survey)
                print('step', nb_iter, ':mobile error=', train_mobile)
                print('step', nb_iter, ':sensor error=', train_sensor)
                print('step', nb_iter, ':total error=', train_total)
                # print(np.round(sess.run(est_rou),2))
                t1 = datetime.datetime.now()
                print('\n', 'CPU time:', t1 - t0, '\n')
        output_link_flow = sess.run(est_v)
        output_path_flow = sess.run(est_f)
        output_path_proportion = sess.run(est_rou1)
        output_od_flow = sess.run(est_q)
        output_od_distribution = sess.run(est_gamma1)
        output_ozone_generation = sess.run(est_alpha)

    tt1 = datetime.datetime.now()
    print('\n', 'CPU time:', tt1 - tt0, '\n')
    ##output the estimation results and loss
    ##save the estimation result of ozone,od,path,link layer
    #ozone
    df_dict = {'ozone_id': ozone_df['ozone_id'].values.tolist(),
               'node_id': ozone_df['o_node_id'].values.tolist(),
               'estimated_alpha': output_ozone_generation.flatten().tolist(),
               'target_alpha': ozone_df['trip_generation'].values.tolist()}
    df = pd.DataFrame(df_dict)
    df = df[['ozone_id','node_id', 'estimated_alpha', 'target_alpha']]
    df.to_csv(outputLocation + 'output_zone_alpha.csv', index=None)
    #od

    df_dict = {'od_id': od_df['od_id'].values.tolist(),
               'o_zone_id': od_df['o_zone_id'].values.tolist(),
               'd_zone_id': od_df['d_zone_id'].values.tolist(),
               'estimated_gamma': output_od_distribution.flatten().tolist(),
               'target_gamma': od_df['OD_split'].values.tolist()}
    df = pd.DataFrame(df_dict)
    df = df[['od_id','o_zone_id', 'd_zone_id', 'estimated_gamma', 'target_gamma']]
    df.to_csv(outputLocation + 'output_od_gamma.csv', index=None)
    #path
    df_dict = {'path_id': path_df['path_id'].values.tolist(),
               'o_zone_id': path_df['o_zone_id'].values.tolist(),
               'd_zone_id': path_df['d_zone_id'].values.tolist(),
               'node_sequence': path_df['node_sequence'].values.tolist(),
               'estimated_proportion': output_path_proportion.flatten().tolist(),
               'target_proportion':path_df['path_proportion'].values.tolist()}
    df = pd.DataFrame(df_dict)
    df = df[['path_id','o_zone_id', 'd_zone_id',  'estimated_proportion', 'target_proportion']]
    df.to_csv(outputLocation + 'output_path_proportion.csv', index=None)
    #link
    df_dict = {'link_id': link_df['link_id'].values.tolist(),
               'o_node_id': link_df['o_node_id'].values.tolist(),
               'd_node_id': link_df['d_node_id'].values.tolist(),
               'estimated_count': output_link_flow.flatten().tolist(),
               'target_count': link_df['sensor_count'].values.tolist()}
    df = pd.DataFrame(df_dict)
    df = df[['link_id', 'o_node_id', 'd_node_id', 'estimated_count', 'target_count']]
    df.to_csv(outputLocation + 'output_link_count.csv', index=None)


    ##loss
    dataframe = pd.DataFrame({'loss_total': list_total, 'loss_survey': list_survey, 'loss_mobile': list_mobile, 'loss_sensor': list_sensor})
    dataframe.to_csv(outputLocation+"output_loss.csv", index=True)

    import matplotlib
    import matplotlib.pyplot as plt

    list_all = [list_survey, list_mobile, list_sensor, list_total]
    c = ['r-', 'b-', 'k', 'y-']
    name_list = ['survey_loss', 'mobile_loss', 'sensor_loss', 'total_loss']
    plt.figure(figsize=(18, 3.5))
    for i in range(4):
        subplot = plt.subplot(1, 4, i + 1)
        subplot.plot(list_all[i], c[i])
        # subplot.axis('tight')
        # subplot.set_xlabel("epoches" )
        # subplot.set_ylabel("loss value")
        plt.title(name_list[i])
        # plt.grid()
    plt.savefig(picLocation + 'loss-BTCGLite.png', dpi=300, format='png')
    plt.show()

