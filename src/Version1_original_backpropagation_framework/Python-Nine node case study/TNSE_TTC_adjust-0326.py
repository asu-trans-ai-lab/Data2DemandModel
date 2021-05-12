
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import time
import datetime
import random


# In[2]:


# Data
class DNode:
    def __init__(self, node_id, is_zone, is_source_zone, generation, population, trip_rate):
        self.node_id = node_id
        self.is_zone = is_zone
        self.is_source_zone = is_source_zone
        self.generation = generation
        self.population = population
        self.trip_rate = trip_rate
        
class DOD:
    def __init__(self, od_id, from_zone_id, to_zone_id, split_ratio, gamma, theta):
        self.od_id = od_id
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.split_ratio = split_ratio
        self.gamma = gamma
        self.theta = theta
        
class DPath:
    def __init__(self, path_id, from_zone_id, to_zone_id, node_sequence, link_sequence, proportion, cost, travel_time):
        self.path_id = path_id
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.node_sequence = node_sequence
        self.link_sequence = link_sequence
        self.proportion = proportion
        self.cost = cost
        self.travel_time = travel_time

class DLink:
    def __init__(self, link_id, from_node_id, to_node_id, length, is_observed,
                 sensor_id=None, travel_time=None, toll=None, flow=0):
        self.link_id = link_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.length = length
        self.is_observed = is_observed
        self.sensor_id = sensor_id
        self.travel_time = travel_time
        self.toll = toll
        self.flow = flow


# In[3]:


# Graph
class Node:
    def __init__(self, dnode):
        self.node_id = dnode.node_id
        self.is_zone = dnode.is_zone
        self.is_source_zone = dnode.is_source_zone
        self.total_gamma_ = None
        self.assume_generation = dnode.generation+random.random()*60-30 if dnode.generation > 0 else 0                ######
        self.alpha_ = tf.Variable(self.assume_generation, dtype=tf.float32, name='Node/alpha/node_'+str(self.node_id))######
        self.PopTR = tf.placeholder(tf.float32,shape=None,name='Node/PopxTR/node_'+str(self.node_id))

        
class OD:
    def __init__(self, dod):
        self.od_id = dod.od_id
        self.from_zone_id = dod.from_zone_id
        self.to_zone_id = dod.to_zone_id
        self.split = tf.placeholder(tf.float32,shape=None,name='OD/split/OD_'+str(self.od_id))
        self.gamma_ = None
        self.gamma_norm_ = tf.Variable(dod.split_ratio+random.random()*0.05, dtype=tf.float32, name='OD/gamma/OD_'+str(self.od_id))
        self.q_ = None
        self.theta_ = tf.Variable(-1.0, dtype=tf.float32, name='OD/theta/OD_'+str(self.od_id))
        self.exp_reci_ = None
        self.including_paths = []
        
class Path:
    def __init__(self, dpath):
        self.path_id = dpath.path_id
        self.from_zone_id = dpath.from_zone_id
        self.to_zone_id = dpath.to_zone_id
        self.node_sequence = dpath.node_sequence
        ## TODO1: use logit model
        self.rou_ = tf.Variable(dpath.proportion, dtype=tf.float32, name='Path/rou/Path_'+str(self.path_id))
        self.t = tf.placeholder(tf.float32,shape=None,name='Path/time/Path_'+str(self.path_id))
        self.flow_ = None
        self.exp_ = None
        self.belonged_od = None

class Link:
    def __init__(self, dlink):
        self.link_id = dlink.link_id
        self.from_node_id = dlink.from_node_id
        self.to_node_id = dlink.to_node_id
        self.is_observed = dlink.is_observed
        self.count = tf.placeholder(tf.float32,shape=None,name='Link/count/link_'+str(self.link_id))
        self.v_ = None
        self.belonged_paths = []


# In[4]:


def load_data():
    print('Preparing Data...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    t0 = datetime.datetime.now()
    node_df = pd.read_csv('input_node.csv')
    zone_df = pd.read_csv('input_zone.csv')
    od_df = pd.read_csv('input_od.csv')
    path_df = pd.read_csv('input_path_set.csv')
    link_df = pd.read_csv('input_link.csv',encoding='gbk')
    
    # prepare for assume data
    g_dict = od_df[['origin','demand']].groupby('origin').sum()['demand']            
    od_df['split_ratio'] = od_df.apply(lambda x: x.demand/g_dict[x.origin], axis=1)  
    od_df['pair'] = od_df.apply(lambda x: (x.origin, x.destination), axis=1)         
    demand_dict = od_df[['pair','demand']].set_index('pair').to_dict()['demand']     
    link_df['pair'] = link_df.apply(lambda x: (x.from_node_id, x.to_node_id), axis=1)
    link_dict = link_df[['pair','link_id']].set_index('pair').to_dict()['link_id']
    
    ### 01-15
    flow_dict = link_df[['link_id','sensor_count']].set_index('link_id').to_dict()['sensor_count']
    
    index_sensor_count = list() 
    for link_id in flow_dict.keys():
        if flow_dict[link_id]!=0:
            index_sensor_count.append(link_id)
        
    for i in range(len(path_df)):
        path_r = path_df.loc[i] 
        od_pair = (path_r.origin_node_id,path_r.destination_node_id)
        path_flow = path_r.path_proportion * demand_dict[od_pair]
        node_list = path_r.path_node_sequence[:-1].split(';')    
        for i in range(1,len(node_list)):
            link_id = link_dict[(int(node_list[i-1]),int(node_list[i]))]
            if link_id not in index_sensor_count:   
                flow_dict[link_id] += path_flow
                
    # Node
    zone_idx = set(zone_df.zone_id)
    source_zones_ids = set(pd.read_csv('input_od.csv').origin)
    node_df['is_zone'] = node_df.apply(lambda x: True if x.node_id in zone_idx else False, axis=1)
    dnode = node_df.apply(lambda x: DNode(x.node_id,
                                            is_zone=x.is_zone, 
                                            is_source_zone=True if x.node_id in source_zones_ids else False,
                                            generation=zone_df[zone_df.zone_id==x.node_id].population.values[0] if x.node_id in zone_idx else 0, 
                                            population=None, 
                                            trip_rate=None), axis=1)

    # OD
    od_df['od_id'] = range(len(od_df))
    dod = od_df.apply(lambda x: DOD(x.od_id,
                                            x.origin,
                                            x.destination, 
                                            x.split_ratio, 
                                            gamma=None,
                                            theta=None), axis=1)
    
    # Path
    path_df['path_id'] = range(len(path_df))
    dpath = path_df.apply(lambda x: DPath(x.path_id,
                                            x.origin_node_id,
                                            x.destination_node_id,
                                            x.path_node_sequence[:-1],
                                            x.link_sequence[:-1],
                                            x.path_proportion,
                                            cost=None,
                                            travel_time=x.path_travel_time), axis=1)
    
    # Link
    dlink = link_df.apply(lambda x: DLink(x.link_id, 
                                            x.from_node_id, 
                                            x.to_node_id, 
                                            x.length, 
                                            True if not np.isnan(x.is_observed) else False,    
                                            x.sensor_id if not np.isnan(x.sensor_id) else None,   
                                            travel_time=None, 
                                            toll=None,
                                            flow=flow_dict[x.link_id]), axis=1)
    
    t1 = datetime.datetime.now()
    print('Using Time: ',t1-t0,'\n')
    
    return{'dnode':dnode, 'dod':dod, 'dpath':dpath, 'dlink':dlink}


# In[5]:


def build_loss(data):
    t0 = datetime.datetime.now()
    print('Initializing Graph Tensors...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    node = [Node(dnode) for dnode in data['dnode']]
    od = [OD(dod) for dod in data['dod']]
    path = [Path(dpath) for dpath in data['dpath']]
    link = [Link(dlink) for dlink in data['dlink']]
    t1 = datetime.datetime.now()
    print('Using Time: ',t1-t0,'\n')
    
    print('Building Connections from Node to OD...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    # node layer to od layer
    ## NOTE: simplify due to [each zone only include one node]
    source_zones_ids = set([od_w.from_zone_id for od_w in od])
    source_zones = [node_i for node_i in node if node_i.node_id in source_zones_ids]
    source_zones = sorted(source_zones, key=lambda node_i: node_i.node_id)
    for od_w in od:
        node_i = node[int(od_w.from_zone_id)]
        od_w.q_ = tf.multiply(od_w.gamma_norm_, node_i.alpha_, name='OD/q/OD_'+str(od_w.od_id))
    t2 = datetime.datetime.now()
    print('Using Time: ',t2-t1,'\n')
          
    print('Building Connections from OD to Path...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    # od layer to path layer
    for path_r in path:
        for od_w in od:
            if od_w.from_zone_id == path_r.from_zone_id and od_w.to_zone_id == path_r.to_zone_id:
                path_r.belonged_od = od_w
                od_w.including_paths.append(path_r)
   
    # TODO1: logit model
    for path_r in path:
        ## NOTE: add cost here if needed
        path_r.exp_ = tf.exp(path_r.belonged_od.theta_ * path_r.t)

    for od_w in od:
        od_w.exp_reci_ = tf.reciprocal(tf.reduce_sum(tf.stack([path_r.exp_ for path_r in od_w.including_paths])))

    for path_r in path:
        path_r.rou_ = tf.multiply(path_r.exp_, path_r.belonged_od.exp_reci_, name='Path/rou/Path_'+str(path_r.path_id))
    # TODO1: logit model
  
    
    for path_r in path:    
        path_r.flow_ = tf.multiply(path_r.rou_, path_r.belonged_od.q_, name='Path/flow/Path_'+str(path_r.path_id))
    t3 = datetime.datetime.now()
    print('Using Time: ',t3-t2,'\n')
          
    print('Building Connections from Path to Link...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    # path layer to link layer
    link_dict = dict()
    for link_l in link:
        link_dict[(link_l.from_node_id, link_l.to_node_id)] = link_l
    for path_r in path:
        node_list = path_r.node_sequence.split(';')
         ## TODO: check whether node_list is longer than 2
        for i in range(1,len(node_list)):
            link_l = link_dict[(int(node_list[i-1]),int(node_list[i]))]
            link_l.belonged_paths.append(path_r)

    for link_l in link:
        link_l.v_ = tf.reduce_sum(tf.stack([path_r.flow_ for path_r in link_l.belonged_paths],
                                           name='Link/v_vector/link_'+str(link_l.link_id)),
                                  name='Link/v/link_'+str(link_l.link_id))
    t4 = datetime.datetime.now()
    print('Using Time: ',t4-t3,'\n')
    
    print('Building Loss...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    # build loss and optimize
    ## TODO: confirm the first term of loss, whether only consider source zones (people go out)
    f1 = tf.reduce_sum(tf.stack([tf.pow(tf.subtract(tf.truediv(node_i.alpha_, node_i.PopTR, 
                                                               name='loss/alpha/div'), 
                                                    1, name='loss/alpha/sub'), 
                                        2, name='loss/alpha/term') 
                                 for node_i in source_zones], name='loss/alpha/vector'), 
                       name='loss/alpha/alpha') 
    tf.summary.scalar('loss/alpha/alpha', f1) ####
    
    f2 = tf.reduce_sum(tf.stack([tf.pow(tf.subtract(tf.truediv(od_w.gamma_norm_, od_w.split,
                                                               name='loss/gamma/div'),
                                                    1, name='loss/gamma/sub'),
                                        2, name='loss/gamma/term')
                                 for od_w in od], name='loss/gamma/vector'),
                       name='loss/gamma/gamma')
    tf.summary.scalar('loss/gamma/gamma', f2) ####
    
    f3 = tf.reduce_sum(tf.stack([tf.pow(tf.subtract(tf.truediv(link_l.v_, link_l.count,
                                                               name='loss/flow/div'),
                                                    1, name='loss/flow/sub'),
                                        2, name='loss/flow/term')
                                 for link_l in link if link_l.is_observed], name='loss/flow/vector'),
                       name='loss/flow/flow')
    tf.summary.scalar('loss/flow/flow', f3) ####

    loss = tf.reduce_sum(tf.stack([f1, f2, f3], name='loss/total_loss_vector'), name='loss/total_loss')
    tf.summary.scalar('loss/total_loss_vector', loss) ####
    
    t5 = datetime.datetime.now()
    print('Using Time: ',t5-t4,'\n')
    
    print('Adding moniter...')
    alpha_ = tf.stack([node_i.alpha_ for node_i in node], name='Node/alpha/vector')
    tf.summary.histogram('Node/alpha/vector', alpha_) ####
    
    gamma_ = tf.stack([od_w.gamma_norm_ for od_w in od], name='OD/gamma/vector')
    tf.summary.histogram('OD/gamma/vector', gamma_) ####
    
    q_ = tf.stack([od_w.q_ for od_w in od], name='OD/q/vector')
    tf.summary.histogram('OD/q/vector', q_) ####
    
    rou_ = tf.stack([path_r.rou_ for path_r in path], name='Path/rou/vector')
    tf.summary.histogram('Path/rou/vector', rou_) ####
    
    flow_ = tf.stack([path_r.flow_ for path_r in path], name='Path/flow/vector')
    tf.summary.histogram('Path/flow/vector', flow_) ####
    
    v_ = tf.stack([link_l.v_ for link_l in link], name='Link/v/vector')
    tf.summary.histogram('Link/v/vector', v_ ) ####
    
    v_ob_ = tf.stack([link_l.v_ for link_l in link if link_l.is_observed], name='Link/v_observed/vector')
    tf.summary.histogram('Link/v_observed/vector', v_ob_ ) 
    
    t6 = datetime.datetime.now()
    print('Using Time: ',t6-t5,'\n')
    return f1, f2, f3, loss


# In[6]:


def build_optimizer(f1, learning_rate1, f2, learning_rate2, f3, learning_rate3, loss, learning_rate):
    t0 = datetime.datetime.now()
    print('Building Optimizer...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate1).minimize(f1, name='optimizer1')
    optimizer2 = tf.train.GradientDescentOptimizer(learning_rate2).minimize(f2, name='optimizer2')
    optimizer3 = tf.train.GradientDescentOptimizer(learning_rate3).minimize(f3, name='optimizer3')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, name='optimizer')
    t1 = datetime.datetime.now()
    print('Using Time: ',t1-t0,'\n')
    return optimizer1, optimizer2, optimizer3, optimizer


# In[7]:


def build_graph(data, learning_rate1, learning_rate2, learning_rate3, learning_rate):
    f1, f2, f3, loss = build_loss(data)
    optimizer1, optimizer2, optimizer3, optimizer = build_optimizer(f1, learning_rate1, f2, learning_rate2, f3, learning_rate3, 
                                                                    loss, learning_rate)
    return f1, optimizer1, f2, optimizer2, f3, optimizer3, loss, optimizer


# In[8]:


def feed_data(data, graph):
    dnode = data['dnode']
    dod = data['dod']
    dpath = data['dpath']
    dlink = data['dlink']
    
    feed = dict()
    t0 = datetime.datetime.now()
    print('Filling Node into Feed Dict...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    for node_i in dnode:
        PopTR = graph.get_tensor_by_name('Node/PopxTR/node_'+str(node_i.node_id)+':0')
        feed[PopTR] = node_i.generation

    t1 = datetime.datetime.now()
    print('Using Time: ',t1-t0,'\n')
    
    print('Filling OD into Feed Dict...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    for od_w in dod:
        split = graph.get_tensor_by_name('OD/split/OD_'+str(od_w.od_id)+':0')
        feed[split] = od_w.split_ratio

    t2 = datetime.datetime.now()
    print('Using Time: ',t2-t1,'\n')
    
    print('Filling Path into Feed Dict...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    for path_r in dpath:
        t = graph.get_tensor_by_name('Path/time/Path_'+str(path_r.path_id)+':0')
        feed[t] = path_r.travel_time
        
    t3 = datetime.datetime.now()
    print('Using Time: ',t3-t2,'\n')
    
    print('Filling Link into Feed Dict...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    for link_l in dlink:
        count = graph.get_tensor_by_name('Link/count/link_'+str(link_l.link_id)+':0')
        feed[count] = link_l.flow+1
        
    t4 = datetime.datetime.now()
    print('Using Time: ',t4-t3,'\n')
    
    return feed


# In[9]:


def output_results(sess,data,feed,graph):
    est_ = sess.run(graph.get_tensor_by_name('Node/alpha/vector:0'), feed_dict=feed)
    df_dict = {'zone_id': [node_i.node_id for node_i in data['dnode']],
                     'estimated_alpha': est_,
                     'target_alpha': [node_i.generation for node_i in data['dnode']]}
    df = pd.DataFrame(df_dict)
    df = df[['zone_id','estimated_alpha','target_alpha']]

    df.to_csv('output_zone_alpha.csv', index=None)
    
    est_ = sess.run(graph.get_tensor_by_name('OD/gamma/vector:0'), feed_dict=feed)
    df_dict = {'from_zone_id': [od_w.from_zone_id for od_w in data['dod']],
                     'to_zone_id': [od_w.to_zone_id for od_w in data['dod']],
                     'estimated_gamma': est_,
                     'target_gamma': [od_w.split_ratio for od_w in data['dod']]}
    df = pd.DataFrame(df_dict)
    df = df[['from_zone_id','to_zone_id','estimated_gamma','target_gamma']]
    df.to_csv('output_od_gamma.csv', index=None)
    
    est_ = sess.run(graph.get_tensor_by_name('Path/rou/vector:0'), feed_dict=feed)
    df_dict = {'from_zone_id': [path_r.from_zone_id for path_r in data['dpath']],
               'to_zone_id': [path_r.to_zone_id for path_r in data['dpath']],
               'path_id': [path_r.path_id for path_r in data['dpath']],
               'estimated_proportion': est_,
               'target_proportion': [path_r.path_id for path_r in data['dpath']]}
    df = pd.DataFrame(df_dict)
    df = df[['from_zone_id','to_zone_id','path_id','estimated_proportion','target_proportion']]
    df.to_csv('output_path_proportion.csv', index=None)
    
    est_ = sess.run(graph.get_tensor_by_name('Link/v/vector:0'), feed_dict=feed)
    df_dict = {'link_id': [link_l.link_id for link_l in data['dlink']],
                   'from_node_id': [link_l.from_node_id for link_l in data['dlink']],
                   'to_node_id': [link_l.to_node_id for link_l in data['dlink']],
                     'estimated_count': est_}
    df = pd.DataFrame(df_dict)
    df = df[['link_id','from_node_id','to_node_id','estimated_count']]
    df.to_csv('output_link_count.csv', index=None)


# In[10]:


data = load_data()
f1, optimizer1, f2, optimizer2, f3, optimizer3, loss, optimizer = build_graph(data, learning_rate1=0.0001, learning_rate2=0.01, 
                                                                              learning_rate3=0.01, learning_rate=0.01)
saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log", sess.graph) ####
    merged = tf.summary.merge_all() #### Package computational graph
    
    print('Initializing/Restoring Tensorflow Variables...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    t0 = datetime.datetime.now()
    sess.run(tf.global_variables_initializer())
    t1 = datetime.datetime.now()
    print('Using Time: ',t1-t0,'\n')# 15min30s
    
    print('Saving Graph...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    t2 = datetime.datetime.now()
    print('Using Time: ',t2-t1,'\n')
    
    print('Preparing Feed Data...')
    print('Time Now:',time.strftime('%H:%M:%S',time.localtime(time.time())))
    graph = tf.get_default_graph()
    feed = feed_data(data, graph)
    
    t3 = datetime.datetime.now()
    print('Using Time: ',t3-t2,'\n')
    
    list0 = []
    list1 = []
    list2 = []
    list3 = []
    
    for step in range(100):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) ####
        run_metadata = tf.RunMetadata() ####
        
        train_mse, _ = sess.run([loss, optimizer], feed_dict=feed) ####
        train_mse1, _ = sess.run([f1, optimizer1], feed_dict=feed)
        train_mse2, _ = sess.run([f2, optimizer2], feed_dict=feed)
        train_mse3, _ = sess.run([f3, optimizer3], feed_dict=feed)
        
        list0.append(train_mse)   
        list1.append(train_mse1)
        list2.append(train_mse2)
        list3.append(train_mse3)
        

        dataframe = pd.DataFrame({'loss':list0, 'loss1':list1, 'loss2':list2, 'loss3':list3})   
        dataframe.to_csv("output_loss.csv", index=False)
        
        result = sess.run(merged, feed_dict=feed) #### Run all computational graphs
        writer.add_summary(result, step) ####
        writer.add_run_metadata(run_metadata, 'step{}'.format(step)) ####
        
        print('training step %i train_mse1 =' % (step), train_mse1,';',
              'train_mse2 =', train_mse2,';',
              'train_mse3 =', train_mse3,';',
              'train_mse =', train_mse,';',
              'time_now', time.strftime('%H:%M:%S',time.localtime(time.time())))
    
    output_results(sess,data,feed,graph)

