#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# datetime: 2020/7/7 9:41
# software: PyCharm
# BTCG_GMNS.py

# In[1]: data location
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import datetime
import random
inputLocation ="SiouxFalls network/"
outputLocation = "SiouxFalls network/"+"BTCG/"
picLocation = inputLocation + "pictures/"

# In[2]:Data format: structure
class DD:
    def __init__(self, d_id,destination):
        self.d_id = d_id
        self.destination = destination

# Data
class DNode:
    '''
    '''
    def __init__(self, node_id, is_zone, is_source_zone, generation, population, trip_rate):
        self.node_id = node_id
        self.is_zone = is_zone
        self.is_source_zone = is_source_zone
        self.generation = generation
        self.population = population
        self.trip_rate = trip_rate


class DOD:
    def __init__(self, od_id, o_zone_id, d_zone_id, split_ratio, o_volume,d_volume,cost, gamma, theta):
        self.od_id = od_id
        self.o_zone_id = o_zone_id
        self.d_zone_id = d_zone_id
        self.split_ratio = split_ratio
        self.o_volume=o_volume
        self.d_volume=d_volume
        self.cost=cost
        self.gamma = gamma
        self.theta = theta


class DPath:
    def __init__(self, path_id, o_zone_id, d_zone_id, node_sequence, link_sequence, path_proportion, cost, travel_time):
        self.path_id = path_id
        self.o_zone_id = o_zone_id
        self.d_zone_id = d_zone_id
        self.node_sequence = node_sequence
        self.link_sequence = link_sequence
        self.path_proportion = path_proportion
        self.cost = cost
        self.travel_time = travel_time


class DLink:
    def __init__(self, link_id, o_node_id, d_node_id, length,sensor_count, is_observed,
                 sensor_id=None, travel_time=None, toll=None, flow=0):
        self.link_id = link_id
        self.o_node_id = o_node_id
        self.d_node_id = d_node_id
        self.length = length
        self.sensor_count=sensor_count
        self.is_observed = is_observed
        self.sensor_id = sensor_id
        self.travel_time = travel_time
        self.toll = toll
        self.flow = flow


# In[3]:
# Some calculations are defined based on the original structure
class Node:
    def __init__(self, dnode):
        self.node_id = dnode.node_id
        self.is_zone = dnode.is_zone
        self.is_source_zone = dnode.is_source_zone
        self.total_gamma_ = None
        self.assume_generation = dnode.generation + random.random() *10 -5 if dnode.generation > 0 else 0
        self.alpha_ = tf.Variable(self.assume_generation, dtype=tf.float32,
                                  name='Node/alpha/node_' + str(self.node_id))
        self.PopTR = tf.placeholder(tf.float32, shape=None, name='Node/PopxTR/node_' + str(self.node_id))

class D:
    def __init__(self, dd):
        self.d_id = dd.d_id
        self.destination =dd.destination
        self.including_od = []
        self.g_gamma_ = tf.Variable(-1.0, dtype=tf.float32)
        self.exp_reci_ = None

class OD:
    def __init__(self, dod):
        self.od_id = dod.od_id
        self.o_zone_id = dod.o_zone_id
        self.d_zone_id = dod.d_zone_id
        self.o_volume=dod.o_volume
        self.d_volume=dod.d_volume
        self.cost=dod.cost
        self.split = tf.placeholder(tf.float32, shape=None, name='OD/split/OD_' + str(self.od_id))
        self.gamma_ = None
        self.gamma_norm_ = tf.Variable(dod.split_ratio + random.random()*0.05, dtype=tf.float32,
                                       name='OD/gamma/OD_' + str(self.od_id))

        self.q_ = None
        self.theta_ = tf.Variable(-1.0, dtype=tf.float32)
        self.exp_reci_ = None
        self.exp_ = None
        self.gamma_gravity_=None
        self.including_paths = []
        self.belonged_d = None


class Path:
    def __init__(self, dpath):
        self.path_id = dpath.path_id
        self.o_zone_id = dpath.o_zone_id
        self.d_zone_id = dpath.d_zone_id
        self.node_sequence = dpath.node_sequence
        ## TODO1: use logit model
        self.rou_ = tf.Variable(dpath.path_proportion, dtype=tf.float32, name='Path/rou/Path_' + str(self.path_id))
        self.t = tf.placeholder(tf.float32, shape=None, name='Path/time/Path_' + str(self.path_id))
        self.flow_ = None
        self.exp_ = None
        self.belonged_od = None


class Link:
    def __init__(self, dlink):
        self.link_id = dlink.link_id
        self.o_node_id = dlink.o_node_id
        self.d_node_id = dlink.d_node_id
        self.is_observed = dlink.is_observed
        self.sensor_count=dlink.sensor_count
        self.count = tf.placeholder(tf.float32, shape=None, name='Link/count/link_' + str(self.link_id))
        self.v_ = None
        self.belonged_paths = []


# In[4]: The process of data loading:prepare the data of each layer and return
def load_data():
    print('Preparing Data...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    t0 = datetime.datetime.now()
    all_node_df = pd.read_csv(inputLocation + 'node.csv', encoding='gbk')
    all_link_df = pd.read_csv(inputLocation + 'road_link.csv', encoding='gbk')
    all_agent_df = pd.read_csv(inputLocation + 'agent.csv', encoding='gbk')
    agent_type_df = pd.read_csv(inputLocation + 'agent_type.csv', encoding='gbk')

    # In[2] Only consider the first sample
    ######The observed values of the four layers are respectively trip_generation,OD_split,path_proportion,sensor_count
    ozone_df = all_agent_df[all_agent_df['agent_type'] == 1]
    ozone_df.rename(columns={'agent_id': 'ozone_id', 'observations': 'trip_generation'}, inplace=True)
    od_df = all_agent_df[all_agent_df['agent_type'] == 2]
    od_df.rename(columns={'agent_id': 'od_id', 'observations': 'OD_split'}, inplace=True)
    path_df = all_agent_df[all_agent_df['agent_type'] == 3]
    path_df.rename(columns={'agent_id': 'path_id', 'observations': 'path_proportion'}, inplace=True)
    link_df = all_agent_df[all_agent_df['agent_type'] == 4]
    link_df.rename(columns={'agent_id': 'link_id', 'observations': 'sensor_count'}, inplace=True)

    ozone_df.reset_index(drop=True, inplace=True)
    od_df.reset_index(drop=True, inplace=True)
    path_df.reset_index(drop=True, inplace=True)
    link_df.reset_index(drop=True, inplace=True)

    # prepare for assume data
    g_dict = od_df[['o_zone_id', 'od_flow']].groupby('o_zone_id').sum()['od_flow']   # calculate the trip generation for different zones
    od_df['split_ratio'] = od_df.apply(lambda x: x.od_flow / g_dict[x.o_zone_id], axis=1)  #  Add column split_ratio in the OD dataframe
    od_df['od_id'] = od_df.apply(lambda x: int(x.od_id), axis=1)

    od_df['pair'] = od_df.apply(lambda x: (x.o_zone_id, x.d_zone_id), axis=1)  # add OD pair
    demand_dict = od_df[['pair', 'od_flow']].set_index('pair').to_dict()['od_flow']  # add OD flow
    link_df['pair'] = link_df.apply(lambda x: (x.o_node_id, x.d_node_id), axis=1)
    link_dict = link_df[['pair', 'link_id']].set_index('pair').to_dict()['link_id']

    ab_dict = od_df[['d_zone_id', 'od_flow']].groupby('d_zone_id').sum()['od_flow']# calculate the trip attraction for different zones
    od_df['o_volume']=od_df.apply(lambda x: g_dict[x.o_zone_id], axis=1)###trafffic ageneration
    od_df['d_volume'] = od_df.apply(lambda x: ab_dict[x.d_zone_id], axis=1)###trafffic attraction
    od_df['cost'] = od_df.apply(lambda x: x.od_flow , axis=1)###cost for every od pair
    # #save the ozone*od array
    kj=0
    for indexs in od_df.index:
        o_zone_id=od_df.loc[indexs]['o_zone_id']
        d_zone_id=od_df.loc[indexs]['d_zone_id']
        path_i_df = path_df[path_df['o_zone_id'] == o_zone_id]
        path_ij_df = path_i_df[path_i_df['d_zone_id'] == d_zone_id]
        cost_ij_value = 0
        for m in path_ij_df.index:
            cost = path_ij_df.loc[m]['path_proportion'] * path_ij_df.loc[m]['time_peroid']
            cost_ij_value += cost
        od_df.loc[indexs]['cost']=cost_ij_value
        kj+=1
    d_df =pd.DataFrame(columns=('d_id','destination'))
    k=0
    for d_zone_id in od_df['d_zone_id'].unique().tolist():
        k = k + 1
        d_id=k
        d_df = d_df.append(pd.DataFrame({'d_id': [d_id], 'destination': [d_zone_id]}), ignore_index=True)



    ### link flow
    flow_dict = link_df[['link_id', 'sensor_count']].set_index('link_id').to_dict()['sensor_count']

    index_sensor_count = list()  # record the link_id which sensor_count!=0
    for link_id in flow_dict.keys():
        if flow_dict[link_id] != 0:
            index_sensor_count.append(link_id)


    ####incomplete sensor data supplementation
    for i in range(len(path_df)):
        path_r = path_df.loc[i]  # Loop out each row in path_df
        od_pair = (path_r.o_zone_id, path_r.d_zone_id)
        path_flow = path_r.path_proportion * demand_dict[od_pair]
        node_list = path_r.node_sequence.split('; ')  # Take all the elements and separated
        for i in range(1, len(node_list)):
            link_id = link_dict[(int(node_list[i - 1]), int(node_list[i]))]
            if link_id not in index_sensor_count:  # supplemenation the sensor_data： path_flow=sum(OD_demand*proportion)
                flow_dict[link_id] += path_flow


    # Node
    zone_idx = set(ozone_df.o_node_id)
    source_zones_ids = set(ozone_df.o_zone_id)
    node_df = all_node_df
    node_df['is_zone'] = node_df.apply(lambda x: True if x.node_id in zone_idx else False, axis=1)
    dnode = node_df.apply(lambda x: DNode(x.node_id,
                                          is_zone=x.is_zone,  #is zone
                                          is_source_zone=True if x.node_id in source_zones_ids else False,  # is ozone
                                          generation=ozone_df[ozone_df.o_node_id == x.node_id].trip_generation.values[
                                              0] if x.node_id in zone_idx else 0,
                                          # two restrictions:1)equal 2)x_node_id in zone_idx
                                          population=None,
                                          trip_rate=None), axis=1)
    # OD
    dd= d_df.apply(lambda x: DD(x.d_id,
                                x.destination
                                ), axis=1)
    # OD
    dod = od_df.apply(lambda x: DOD(x.od_id,
                                    x.o_zone_id,
                                    x.d_zone_id,
                                    x.split_ratio,
                                    x.o_volume,
                                    x.d_volume,
                                    x.cost,
                                    gamma=None,
                                    theta=None), axis=1)

    # Path
    path_df['path_id'] = range(len(path_df))
    dpath = path_df.apply(lambda x: DPath(x.path_id,
                                          x.o_zone_id,
                                          x.d_zone_id,
                                          x.node_sequence,
                                          x.node_sequence,
                                          x.path_proportion,
                                          cost=None,
                                          travel_time=x.time_peroid), axis=1)

    # Link
    link_df['path_id'] = range(len(link_df))
    link_df['sensor_id'] = range(len(link_df))
    dlink = link_df.apply(lambda x: DLink(x.link_id,
                                          x.o_node_id,
                                          x.d_node_id,
                                          x.time_peroid,
                                          x.sensor_count,
                                          True if not np.isnan(x.sensor_id) else False,
                                          x.sensor_id if not np.isnan(x.sensor_id) else None,
                                          travel_time=None,
                                          toll=None,
                                          flow=flow_dict[x.link_id]), axis=1)

    t1 = datetime.datetime.now()
    print('Using Time: ', t1 - t0, '\n')

    return {'dd':dd, 'dnode': dnode, 'dod': dod, 'dpath': dpath, 'dlink': dlink}



# In[5]:define the loss function: f1,f2,f3,f_total
def build_loss(data):
    t0 = datetime.datetime.now()
    print('Initializing Graph Tensors...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))

    d=[D(dd) for dd in data['dd']]
    node = [Node(dnode) for dnode in data['dnode']]
    od = [OD(dod) for dod in data['dod']]
    path = [Path(dpath) for dpath in data['dpath']]
    link = [Link(dlink) for dlink in data['dlink']]
    t1 = datetime.datetime.now()
    print('Using Time: ', t1 - t0, '\n')

    print('Building Connections from Node to OD...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    # node layer to od layer
    ## NOTE: simplify due to [each zone only include one node]
    source_zones_ids = set([od_w.o_zone_id for od_w in od])
    source_zones = [node_i for node_i in node if node_i.node_id in source_zones_ids]
    source_zones = sorted(source_zones, key=lambda node_i: node_i.node_id)

    for od_r in od:
        for d_i in d:
            if d_i.destination == od_r.d_zone_id:
                od_r.belonged_d = d_i
                d_i.including_od.append(od_r)
    # TODO1: gravity model,
    ## NOTE: add cost here if needed
    for od_r in od:

        od_r.exp_ = tf.multiply(tf.cast(od_r.d_volume,tf.float32),tf.pow(tf.cast(od_r.cost,tf.float32),od_r.belonged_d.g_gamma_))

    for d_od in d:
        d_od.exp_reci_ = tf.reciprocal(tf.reduce_sum(tf.stack([od_r.exp_ for od_r in d_od.including_od])))

    for od_r in od:
        od_r.gamma_gravity_ = tf.multiply(od_r.exp_, od_r.belonged_d.exp_reci_,
                                  name='OD/gravity_model/OD_' + str(od_r.od_id))

    for od_w in od:
        node_i = node[int(od_w.o_zone_id-1)]
        od_w.q_ = tf.multiply(od_w.gamma_norm_, node_i.alpha_, name='OD/q/OD_' + str(od_w.od_id))

    t2 = datetime.datetime.now()
    print('Using Time: ', t2 - t1, '\n')

    print('Building Connections from OD to Path...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    # od layer to path layer
    for path_r in path:
        for od_w in od:
            if od_w.o_zone_id == path_r.o_zone_id and od_w.d_zone_id == path_r.d_zone_id:
                path_r.belonged_od = od_w
                od_w.including_paths.append(path_r)

    # TODO1: logit model
    ## NOTE: add cost here if needed
    for path_r in path:
        path_r.exp_ = tf.exp(path_r.belonged_od.theta_ * path_r.t)

    for od_w in od:
        od_w.exp_reci_ = tf.reciprocal(tf.reduce_sum(tf.stack([path_r.exp_ for path_r in od_w.including_paths])))

    for path_r in path:
        path_r.rou_ = tf.multiply(path_r.exp_, path_r.belonged_od.exp_reci_,
                                  name='Path/rou/Path_' + str(path_r.path_id))

    for path_r in path:
        path_r.flow_ = tf.multiply(path_r.rou_, path_r.belonged_od.q_, name='Path/flow/Path_' + str(path_r.path_id))
    t3 = datetime.datetime.now()
    print('Using Time: ', t3 - t2, '\n')

    print('Building Connections from Path to Link...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    # path layer to link layer
    link_dict = dict()
    for link_l in link:
        link_dict[(link_l.o_node_id, link_l.d_node_id)] = link_l
    for path_r in path:
        node_list = path_r.node_sequence.split(';')
        ## TODO: check whether node_list is longer than 2
        for i in range(1, len(node_list)):
            link_l = link_dict[(int(node_list[i - 1]), int(node_list[i]))]
            link_l.belonged_paths.append(path_r)

    for link_l in link:
        link_l.v_ = tf.reduce_sum(tf.stack([path_r.flow_ for path_r in link_l.belonged_paths],
                                           name='Link/v_vector/link_' + str(link_l.link_id)),
                                  name='Link/v/link_' + str(link_l.link_id))
    t4 = datetime.datetime.now()
    print('Using Time: ', t4 - t3, '\n')

    print('Building Loss...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    # build loss and optimize
    ## TODO: confirm the first term of loss, whether only consider source zones (people go out)
    f1 = tf.reduce_mean(tf.stack([tf.pow(tf.subtract(node_i.alpha_, node_i.PopTR), 2) for node_i in source_zones]))

    f2 = tf.reduce_mean(tf.stack([tf.pow(tf.subtract(od_w.gamma_norm_, od_w.split), 2) for od_w in od]))

    f3 = tf.reduce_mean(tf.stack(
        [tf.pow(tf.subtract(tf.truediv(link_l.v_, link_l.count), 1), 2) for link_l in link if link_l.is_observed]))


    loss = tf.reduce_sum(tf.stack([f1, f2, f3], name='loss/total_loss_vector'), name='loss/total_loss')
    tf.summary.scalar('loss/total_loss_vector', loss)  ####

    t5 = datetime.datetime.now()
    print('Using Time: ', t5 - t4, '\n')

    print('Adding moniter...')
    alpha_ = tf.stack([node_i.alpha_ for node_i in node], name='Node/alpha/vector')
    tf.summary.histogram('Node/alpha/vector', alpha_)

    gamma_ = tf.stack([od_w.gamma_norm_ for od_w in od], name='OD/gamma/vector')
    tf.summary.histogram('OD/gamma/vector', gamma_)

    q_ = tf.stack([od_w.q_ for od_w in od], name='OD/q/vector')
    tf.summary.histogram('OD/q/vector', q_)

    rou_ = tf.stack([path_r.rou_ for path_r in path], name='Path/rou/vector')
    tf.summary.histogram('Path/rou/vector', rou_)

    flow_ = tf.stack([path_r.flow_ for path_r in path], name='Path/flow/vector')
    tf.summary.histogram('Path/flow/vector', flow_)

    v_ = tf.stack([link_l.v_ for link_l in link], name='Link/v/vector')
    tf.summary.histogram('Link/v/vector', v_)

    v_ob_ = tf.stack([link_l.v_ for link_l in link if link_l.is_observed], name='Link/v_observed/vector')
    tf.summary.histogram('Link/v_observed/vector', v_ob_)

    t6 = datetime.datetime.now()
    print('Using Time: ', t6 - t5, '\n')
    return f1, f2, f3, loss


# In[6]: define the optimizer:optimizer1, optimizer2, optimizer3, optimizer
def build_optimizer(f1, learning_rate1, f2, learning_rate2, f3, learning_rate3, loss, learning_rate):
    t0 = datetime.datetime.now()
    print('Building Optimizer...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate1).minimize(f1, name='optimizer1')
    optimizer2 = tf.train.GradientDescentOptimizer(learning_rate2).minimize(f2, name='optimizer2')
    optimizer3 = tf.train.GradientDescentOptimizer(learning_rate3).minimize(f3, name='optimizer3')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name='optimizer')
    t1 = datetime.datetime.now()
    print('Using Time: ', t1 - t0, '\n')
    return optimizer1, optimizer2, optimizer3, optimizer


# In[7]: build the computation graph
def build_graph(data, learning_rate1, learning_rate2, learning_rate3, learning_rate):
    f1, f2, f3, loss = build_loss(data)
    optimizer1, optimizer2, optimizer3, optimizer = build_optimizer(f1, learning_rate1, f2, learning_rate2, f3,
                                                                    learning_rate3,
                                                                    loss, learning_rate)
    return f1, optimizer1, f2, optimizer2, f3, optimizer3, loss, optimizer


# In[8]:fill the placeholder
def feed_data(data, graph):
    dnode = data['dnode']
    dod = data['dod']
    dpath = data['dpath']
    dlink = data['dlink']

    feed = dict()
    t0 = datetime.datetime.now()
    print('Filling Node into Feed Dict...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    for node_i in dnode:
        PopTR = graph.get_tensor_by_name('Node/PopxTR/node_' + str(node_i.node_id) + ':0')
        feed[PopTR] = node_i.generation

    t1 = datetime.datetime.now()
    print('Using Time: ', t1 - t0, '\n')

    print('Filling OD into Feed Dict...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    for od_w in dod:
        split = graph.get_tensor_by_name('OD/split/OD_' + str(od_w.od_id) + ':0')
        feed[split] = od_w.split_ratio

    t2 = datetime.datetime.now()
    print('Using Time: ', t2 - t1, '\n')

    print('Filling Path into Feed Dict...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    for path_r in dpath:
        t = graph.get_tensor_by_name('Path/time/Path_' + str(path_r.path_id) + ':0')
        feed[t] = path_r.travel_time

    t3 = datetime.datetime.now()
    print('Using Time: ', t3 - t2, '\n')

    print('Filling Link into Feed Dict...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    for link_l in dlink:
        count = graph.get_tensor_by_name('Link/count/link_' + str(link_l.link_id) + ':0')
        feed[count] = link_l.flow + 1

    t4 = datetime.datetime.now()
    print('Using Time: ', t4 - t3, '\n')

    return feed


# In[9]: output the results
def output_results(sess, data, feed, graph):
    est_ = sess.run(graph.get_tensor_by_name('Node/alpha/vector:0'), feed_dict=feed)

    df_dict = {'zone_id': [node_i.node_id for node_i in data['dnode']],
               'node_id': [node_i.node_id for node_i in data['dnode']],
               'estimated_alpha': est_,
               'target_alpha': [node_i.generation for node_i in data['dnode']]}
    df = pd.DataFrame(df_dict)
    df = df[['zone_id', 'node_id','estimated_alpha', 'target_alpha']]

    df.to_csv(outputLocation+'output_zone_alpha.csv', index=None)

    est_ = sess.run(graph.get_tensor_by_name('OD/gamma/vector:0'), feed_dict=feed)
    df_dict = {'od_id': [od_w.od_id for od_w in data['dod']],
               'o_zone_id': [od_w.o_zone_id for od_w in data['dod']],
               'd_zone_id': [od_w.d_zone_id for od_w in data['dod']],
               'estimated_gamma': est_,
               'target_gamma': [od_w.split_ratio for od_w in data['dod']]}
    df = pd.DataFrame(df_dict)
    df = df[['od_id','o_zone_id', 'd_zone_id', 'estimated_gamma', 'target_gamma']]
    df.to_csv(outputLocation+'output_od_gamma.csv', index=None)

    est_ = sess.run(graph.get_tensor_by_name('Path/rou/vector:0'), feed_dict=feed)
    df_dict = {'path_id': [path_r.path_id for path_r in data['dpath']],
               'o_zone_id': [path_r.o_zone_id for path_r in data['dpath']],
               'd_zone_id': [path_r.d_zone_id for path_r in data['dpath']],
               'estimated_proportion': est_,
               'target_proportion': [path_r.path_proportion for path_r in data['dpath']]}
    df = pd.DataFrame(df_dict)
    df = df[['path_id','o_zone_id', 'd_zone_id',  'estimated_proportion', 'target_proportion']]
    df.to_csv(outputLocation+'output_path_proportion.csv', index=None)

    est_ = sess.run(graph.get_tensor_by_name('Link/v/vector:0'), feed_dict=feed)
    df_dict = {'link_id': [link_l.link_id for link_l in data['dlink']],
               'o_node_id': [link_l.o_node_id for link_l in data['dlink']],
               'd_node_id': [link_l.d_node_id for link_l in data['dlink']],
               'estimated_count': est_,
               'target_count':[link_l.sensor_count for link_l in data['dlink']]}
    df = pd.DataFrame(df_dict)
    df = df[['link_id', 'o_node_id', 'd_node_id', 'estimated_count','target_count']]
    df.to_csv(outputLocation+'output_link_count.csv', index=None)


# In[10]:training
data = load_data()
maximum_iterations=1001
curr_iter = tf.Variable(0)  # Current iteration times
# learning rate exponential decay
lr_survey = tf.train.exponential_decay(0.5, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)
lr_mobile = tf.train.exponential_decay(0.00001, curr_iter, decay_steps=1 , decay_rate=0.99)
lr_count = tf.train.exponential_decay(0.005, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)
learning_rate = tf.train.exponential_decay(1e-3, curr_iter, decay_steps=maximum_iterations, decay_rate=0.99)

f1, optimizer1, f2, optimizer2, f3, optimizer3, loss, optimizer= build_graph(data,lr_survey,lr_mobile, lr_count,learning_rate)
saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log", sess.graph)
    merged = tf.summary.merge_all()  #### Package computational graph

    print('Initializing/Restoring Tensorflow Variables...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    t0 = datetime.datetime.now()
    sess.run(tf.global_variables_initializer())
    t1 = datetime.datetime.now()
    print('Using Time: ', t1 - t0, '\n')

    print('Saving Graph...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    t2 = datetime.datetime.now()
    print('Using Time: ', t2 - t1, '\n')

    print('Preparing Feed Data...')
    print('Time Now:', time.strftime('%H:%M:%S', time.localtime(time.time())))
    graph = tf.get_default_graph()
    feed = feed_data(data, graph)

    t3 = datetime.datetime.now()
    print('Using Time: ', t3 - t2, '\n')

    # build list for loss record
    list_survey = []
    list_mobile = []
    list_sensor = []
    list_total = []

    for step in range(maximum_iterations):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        train_mse, _ = sess.run([loss, optimizer], feed_dict=feed)
        train_mse1, _ = sess.run([f1, optimizer1], feed_dict=feed)
        train_mse2, _ = sess.run([f2, optimizer2], feed_dict=feed)
        train_mse3, _ = sess.run([f3, optimizer3], feed_dict=feed)

        list_total.append(train_mse)
        list_survey.append(train_mse1)
        list_mobile.append(train_mse2)
        list_sensor.append(train_mse3)

        result = sess.run(merged, feed_dict=feed)  #### Run all computational graphs
        writer.add_summary(result, step)
        writer.add_run_metadata(run_metadata, 'step{}'.format(step))

        print('training step %i train_mse1 =' % (step), train_mse1, ';',
              'train_mse2 =', train_mse2, ';',
              'train_mse3 =', train_mse3, ';',
              'train_mse =', train_mse, ';',
              'time_now', time.strftime('%H:%M:%S', time.localtime(time.time())))

    ##output the estimation results and loss
    output_results(sess, data, feed, graph)
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
    plt.savefig(picLocation + 'loss-BTCG.png', dpi=300, format='png')
    plt.show()






