import tensorflow as tf
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import networkx as nx

file = open("/home/tingyi/Distinct_embedding/data/full_team_members_year_c_0rmd.txt")
file2 = open("/home/tingyi/Distinct_embedding/data/merged_DiversityData2_0rmd.txt")

"""
Create dictionary of author to team
"""
dic_author = {}
for line in file:
    length = np.array(line.split('\t')).shape[0]
    a = np.int(np.array(line.split('\t'))[0])
    author_id = np.int(np.array(line.split('\t'))[4])
    dic_author.setdefault(author_id, []).append(a)
    # dic_author[author_id]=a
    if length == 5:
        author_id2 = 0
        author_id3 = 0
        author_id4 = 0
        author_id5 = 0
    if length == 6:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        dic_author.setdefault(author_id2, []).append(a)
        author_id3 = 0
        author_id4 = 0
        author_id5 = 0
    if length == 7:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        author_id3 = np.int(np.array(line.split('\t'))[6])
        author_id4 = 0
        author_id5 = 0
        dic_author.setdefault(author_id2, []).append(a)
        dic_author.setdefault(author_id3, []).append(a)
    if length == 8:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        author_id3 = np.int(np.array(line.split('\t'))[6])
        author_id4 = np.int(np.array(line.split('\t'))[7])
        author_id5 = 0
        dic_author.setdefault(author_id2, []).append(a)
        dic_author.setdefault(author_id3, []).append(a)
        dic_author.setdefault(author_id4, []).append(a)
    if length == 9:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        author_id3 = np.int(np.array(line.split('\t'))[6])
        author_id4 = np.int(np.array(line.split('\t'))[7])
        author_id5 = np.int(np.array(line.split('\t'))[8])
        dic_author.setdefault(author_id2, []).append(a)
        dic_author.setdefault(author_id3, []).append(a)
        dic_author.setdefault(author_id4, []).append(a)
        dic_author.setdefault(author_id5, []).append(a)
    if length > 9:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        author_id3 = np.int(np.array(line.split('\t'))[6])
        author_id4 = np.int(np.array(line.split('\t'))[7])
        author_id5 = np.int(np.array(line.split('\t'))[8])
        dic_author.setdefault(author_id2, []).append(a)
        dic_author.setdefault(author_id3, []).append(a)
        dic_author.setdefault(author_id4, []).append(a)
        dic_author.setdefault(author_id5, []).append(a)
file = open("/home/tingyi/Distinct_embedding/data/full_team_members_year_c_0rmd.txt")
file2 = open("/home/tingyi/Distinct_embedding/data/merged_DiversityData2_0rmd.txt")

"""
Create dictionary of team to author
"""
dic_team = {}
for line in file:
    length = np.array(line.split('\t')).shape[0]
    a = np.int(np.array(line.split('\t'))[0])
    author_id = np.int(np.array(line.split('\t'))[4])
    # dic_team[a]=[author_id]
    if length == 5:
        author_id2 = 0
        author_id3 = 0
        dic_team[a] = [author_id]
    if length == 6:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        dic_team[a] = [author_id, author_id2]
        author_id3 = 0
    if length == 7:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        author_id3 = np.int(np.array(line.split('\t'))[6])
        dic_team[a] = [author_id, author_id2, author_id3]
    if length == 8:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        author_id3 = np.int(np.array(line.split('\t'))[6])
        author_id4 = np.int(np.array(line.split('\t'))[7])
        dic_team[a] = [author_id, author_id2, author_id3, author_id4]
    if length == 9:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        author_id3 = np.int(np.array(line.split('\t'))[6])
        author_id4 = np.int(np.array(line.split('\t'))[7])
        author_id5 = np.int(np.array(line.split('\t'))[8])
        dic_team[a] = [author_id, author_id2, author_id3, author_id4, author_id5]
    if length > 9:
        author_id2 = np.int(np.array(line.split('\t'))[5])
        author_id3 = np.int(np.array(line.split('\t'))[6])
        author_id4 = np.int(np.array(line.split('\t'))[7])
        author_id5 = np.int(np.array(line.split('\t'))[8])
        dic_team[a] = [author_id, author_id2, author_id3, author_id4, author_id5]

"""
Create dictionary of connected team
"""
dic_connect_team = {}
for kk in dic_team.keys():
    author_set = dic_team[kk]
    for author_ind in author_set:
        for team_ind in dic_author[author_ind]:
            dic_connect_team.setdefault(kk, []).append(team_ind)

"""
Create dictionary for team attribute
"""
dic_attribute = {}
for line2 in file2:
    tid = np.int(np.array(line2.split('\t'))[0])
    p_year = np.float(np.array(line2.split('\t'))[1])
    citation = np.float(np.array(line2.split('\t'))[2])
    avg_citation = np.float(np.array(line2.split('\t'))[3])
    country_diversity = np.float(np.array(line2.split('\t'))[4])
    topic_diversity = np.float(np.array(line2.split('\t'))[5])
    productivity_diversity = np.float(np.array(line2.split('\t'))[6])
    impact_diversity = np.float(np.array(line2.split('\t'))[7])
    scientific_age_diversity = np.float(np.array(line2.split('\t'))[8])
    dic_attribute[tid] = [country_diversity, topic_diversity, productivity_diversity, impact_diversity,
                          scientific_age_diversity, tid, p_year, citation]

"""
Create graph
"""
def single_node_connection(G,node):
  neighbors = dic_connect_team[node]
  hop_nodes_one = []
  for tid in neighbors:
    if not G.has_node(tid):
      hop_nodes_one.append(tid)
      country_diversity = dic_attribute[tid][0]
      topic_diversity = dic_attribute[tid][1]
      productivity_diversity = dic_attribute[tid][2]
      impact_diversity = dic_attribute[tid][3]
      scientific_age_diversity = dic_attribute[tid][4]
      tid_ = dic_attribute[tid][5]
      p_year = dic_attribute[tid][6]
      citation = dic_attribute[tid][7]
      if len(dic_team[tid])==1:
        author_id1 = dic_team[tid][0]
        author_id2 = 0
        author_id3 = 0
        author_id4 = 0
        author_id5 = 0
      if len(dic_team[tid])==2:
        author_id1 = dic_team[tid][0]
        author_id2 = dic_team[tid][1]
        author_id3 = 0
        author_id4 = 0
        author_id5 = 0
      if len(dic_team[tid])==3:
        author_id1 = dic_team[tid][0]
        author_id2 = dic_team[tid][1]
        author_id3 = dic_team[tid][2]
        author_id4 = 0
        author_id5 = 0
      if len(dic_team[tid])==4:
        author_id1 = dic_team[tid][0]
        author_id2 = dic_team[tid][1]
        author_id3 = dic_team[tid][2]
        author_id4 = dic_team[tid][3]
        author_id5 = 0
      if len(dic_team[tid])==5:
        author_id1 = dic_team[tid][0]
        author_id2 = dic_team[tid][1]
        author_id3 = dic_team[tid][2]
        author_id4 = dic_team[tid][3]
        author_id5 = dic_team[tid][4]
      G.add_node(tid,country_diversity=country_diversity)
      G.add_node(tid,topic_diversity=topic_diversity)
      G.add_node(tid,productivity_diversity=productivity_diversity)
      G.add_node(tid,impact_diversity=impact_diversity)
      G.add_node(tid,scientific_age_diversity=scientific_age_diversity)
      #G.add_node(tid,node_index=node_index)
      G.add_node(tid,tid=tid_)
      G.add_node(tid,p_year=p_year)
      G.add_node(tid,citation=citation)
      G.add_node(tid,author_id1=author_id1)
      G.add_node(tid,author_id2=author_id2)
      G.add_node(tid,author_id3=author_id3)
      G.add_node(tid,author_id4=author_id4)
      G.add_node(tid,author_id5=author_id5)
    G.add_edge(tid,node)
    G.add_edge(node,tid)
  return hop_nodes_one

def create_nodes_one_hop(G,hop_nodes):
  for nodes in hop_nodes:
    single_node_connection(G,nodes)

G = nx.DiGraph()
tid = list(dic_attribute.keys())[0]
country_diversity = dic_attribute[tid][0]
topic_diversity = dic_attribute[tid][1]
productivity_diversity = dic_attribute[tid][2]
impact_diversity = dic_attribute[tid][3]
scienctific_age_diversity = dic_attribute[tid][4]
tid_ = dic_attribute[tid][5]
p_year = dic_attribute[tid][6]
citation = dic_attribute[tid][7]
if len(dic_team[tid])==1:
  author_id1 = dic_team[tid][0]
  author_id2 = 0
  author_id3 = 0
  author_id4 = 0
  author_id5 = 0
if len(dic_team[tid])==2:
  author_id1 = dic_team[tid][0]
  author_id2 = dic_team[tid][1]
  author_id3 = 0
  author_id4 = 0
  author_id5 = 0
if len(dic_team[tid])==3:
  author_id1 = dic_team[tid][0]
  author_id2 = dic_team[tid][1]
  author_id3 = dic_team[tid][2]
  author_id4 = 0
  author_id5 = 0
if len(dic_team[tid])==4:
  author_id1 = dic_team[tid][0]
  author_id2 = dic_team[tid][1]
  author_id3 = dic_team[tid][2]
  author_id4 = dic_team[tid][3]
  author_id5 = 0
if len(dic_team[tid])==5:
  author_id1 = dic_team[tid][0]
  author_id2 = dic_team[tid][1]
  author_id3 = dic_team[tid][2]
  author_id4 = dic_team[tid][3]
  author_id5 = dic_team[tid][4]
G.add_node(tid,country_diversity=country_diversity)
G.add_node(tid,topic_diversity=topic_diversity)
G.add_node(tid,productivity_diversity=productivity_diversity)
G.add_node(tid,impact_diversity=impact_diversity)
G.add_node(tid,scientific_age_diversity=scientific_age_diversity)
#G.add_node(tid,node_index=node_index)
G.add_node(tid,tid=tid_)
G.add_node(tid,p_year=p_year)
G.add_node(tid,citation=citation)
G.add_node(tid,author_id1=author_id1)
G.add_node(tid,author_id2=author_id2)
G.add_node(tid,author_id3=author_id3)
G.add_node(tid,author_id4=author_id4)
G.add_node(tid,author_id5=author_id5)

current_hops = [tid]
hop_next = []
hop_index = 0
hop_size = 10
while(hop_index<4):
  for node_cur in current_hops:
    hop_one = single_node_connection(G,node_cur)
    hop_next = hop_next + hop_one
  current_hops = hop_next
  print(current_hops)
  hop_next = []
  hop_index += 1

index = 0
for node in G.nodes():
  G.add_node(node,node_index=index)
  index += 1

"""
Compute mean for attributes, for later normalization
"""
#mean_year = np.mean([G.nodes[k]['p_year'] for k in G.nodes()])
#mean_cit = np.mean([G.nodes[k]['citation'] for k in G.nodes()])
#mean_avg = np.mean([G.nodes[k]['avg_citation'] for k in G.nodes()])
mean_count = np.mean([G.node[k]['country_diversity'] for k in G.nodes()])
mean_top = np.mean([G.node[k]['topic_diversity'] for k in G.nodes()])
mean_prod = np.mean([G.node[k]['productivity_diversity'] for k in G.nodes()])
mean_impact = np.mean([G.node[k]['impact_diversity'] for k in G.nodes()])
mean_sci = np.mean([G.node[k]['scientific_age_diversity'] for k in G.nodes()])
"""
Compute min and max for attributes
"""
#min_year = np.min([G.nodes[k]['p_year'] for k in G.nodes()])
#min_cit = np.min([G.nodes[k]['citation'] for k in G.nodes()])
#min_avg = np.min([G.nodes[k]['avg_citation'] for k in G.nodes()])
min_count = np.min([G.node[k]['country_diversity'] for k in G.nodes()])
min_top = np.min([G.node[k]['topic_diversity'] for k in G.nodes()])
min_prod = np.min([G.node[k]['productivity_diversity'] for k in G.nodes()])
min_impact = np.min([G.node[k]['impact_diversity'] for k in G.nodes()])
min_sci = np.min([G.node[k]['scientific_age_diversity'] for k in G.nodes()])

#max_year = np.max([G.nodes[k]['p_year'] for k in G.nodes()])
#max_cit = np.max([G.nodes[k]['citation'] for k in G.nodes()])
#max_avg = np.max([G.nodes[k]['avg_citation'] for k in G.nodes()])
max_count = np.max([G.node[k]['country_diversity'] for k in G.nodes()])
max_top = np.max([G.node[k]['topic_diversity'] for k in G.nodes()])
max_prod = np.max([G.node[k]['productivity_diversity'] for k in G.nodes()])
max_impact = np.max([G.node[k]['impact_diversity'] for k in G.nodes()])
max_sci = np.max([G.node[k]['scientific_age_diversity'] for k in G.nodes()])

"""
compute std for attribute
"""
#std_year = np.std([G.nodes[k]['p_year'] for k in G.nodes()])
#std_cit = np.std([G.nodes[k]['citation'] for k in G.nodes()])
#std_avg = np.std([G.nodes[k]['avg_citation'] for k in G.nodes()])
std_count = np.std([G.node[k]['country_diversity'] for k in G.nodes()])
std_top = np.std([G.node[k]['topic_diversity'] for k in G.nodes()])
std_prod = np.std([G.node[k]['productivity_diversity'] for k in G.nodes()])
std_impact = np.std([G.node[k]['impact_diversity'] for k in G.nodes()])
std_sci = np.std([G.node[k]['scientific_age_diversity'] for k in G.nodes()])

"""
Compute derived min&max statistic
"""
#std_max_year = mean_year + std_year
#std_max_cit = mean_cit + std_cit
#std_max_avg = mean_avg + std_avg
std_max_count = mean_count + std_count
std_max_top = mean_top + std_top
std_max_prod = mean_prod + std_prod
std_max_impact = mean_impact + std_impact
std_max_sci = mean_sci + std_sci

#std_min_year = mean_year - std_year
#std_min_cit = mean_cit - std_cit
#std_min_avg = mean_avg - std_avg
std_min_count = mean_count - std_count
std_min_top = mean_top - std_top
std_min_prod = mean_prod - std_prod
std_min_impact = mean_impact - std_impact
std_min_sci = mean_sci - std_sci

"""
Get attribute data
"""


def get_attribute_data_skip_gram(node_index, walk_length):
    return Node2vec.node2vec_walk(walk_length, node_index)


"""
Assign values to attritbute vector
"""


def assign_value(G, node_index):
    attribute_vector = np.zeros(5)
    """
    attribute_vector[0] = np.float(1/(1+np.exp(-(G.nodes[node_index]['ageDiversity']-mean_age))))
    attribute_vector[1] = np.float(1/(1+np.exp(-(G.nodes[node_index]['an_citation']-mean_an))))
    attribute_vector[2] = np.float(1/(1+np.exp(-(G.nodes[node_index]['citationDiversity']-mean_cit))))
    attribute_vector[3] = np.float(1/(1+np.exp(-(G.nodes[node_index]['publicationDiversity']-mean_pub))))
    attribute_vector[4] = np.float(1/(1+np.exp(-(G.nodes[node_index]['topicDiversity']-mean_top))))
    """
    """
    attribute_vector[0] = np.float((G.nodes[node_index]['p_year']-min_year)/(max_year-min_year))
    attribute_vector[1] = np.float((G.nodes[node_index]['citation']-min_cit)/(max_cit-min_cit))
    attribute_vector[2] = np.float((G.nodes[node_index]['avg_citation']-min_avg)/(max_avg-min_avg))
    attribute_vector[3] = np.float((G.nodes[node_index]['country_diversity']-min_count)/(max_count-min_count))
    attribute_vector[4] = np.float((G.nodes[node_index]['topic_diversity']-min_top)/(max_top-min_top))
    attribute_vector[5] = np.float((G.nodes[node_index]['productivity_diversity']-min_prod)/(max_prod-min_prod))
    attribute_vector[6] = np.float((G.nodes[node_index]['impact_diversity']-min_impact)/(max_impact-min_impact))
    attribute_vector[7] = np.float((G.nodes[node_index]['scientific_age_diversity']-min_sci)/(max_sci-min_sci))
    """
    """
    if np.float(G.nodes[node_index]['p_year']) > std_max_year:
      attribute_vector[0] = 1
    elif np.float(G.nodes[node_index]['p_year']) < std_min_year:
      attribute_vector[0] = 0
    else:
      attribute_vector[0] = (np.float(G.nodes[node_index]['p_year'])-std_min_year)/(2*std_year)
  
    if np.float(G.nodes[node_index]['citation']) > std_max_cit:
      attribute_vector[1] = 1
    elif np.float(G.nodes[node_index]['citation']) < std_min_cit:
      attribute_vector[1] = 0
    else:
      attribute_vector[1] = (np.float(G.nodes[node_index]['citation'])-std_min_cit)/(2*std_cit)
  
    if np.float(G.nodes[node_index]['avg_citation']) > std_max_avg:
      attribute_vector[2] = 1
    elif np.float(G.nodes[node_index]['avg_citation']) < std_min_avg:
      attribute_vector[2] = 0
    else:
      attribute_vector[2] = (np.float(G.nodes[node_index]['avg_citation'])-std_min_avg)/(2*std_avg)
    """

    # if np.float(G.nodes[node_index]['country_diversity']) > std_max_count:
    # attribute_vector[0] = 1
    # elif np.float(G.nodes[node_index]['country_diversity']) < std_min_count:
    # attribute_vector[0] = 0
    # else:
    attribute_vector[0] = (np.float(G.node[node_index]['country_diversity']) - mean_count) / std_count

    # if np.float(G.nodes[node_index]['topic_diversity']) > std_max_top:
    # attribute_vector[4] = 1
    # elif np.float(G.nodes[node_index]['topic_diversity']) < std_min_top:
    # attribute_vector[4] = 0
    # else:
    attribute_vector[1] = (np.float(G.node[node_index]['topic_diversity']) - mean_top) / std_top

    # if np.float(G.nodes[node_index]['productivity_diversity']) > std_max_prod:
    # attribute_vector[5] = 1
    # elif np.float(G.nodes[node_index]['productivity_diversity']) < std_min_prod:
    #  attribute_vector[5] = 0
    # else:
    attribute_vector[2] = (np.float(G.node[node_index]['productivity_diversity']) - mean_prod) / std_prod

    # if np.float(G.nodes[node_index]['impact_diversity']) > std_max_impact:
    # attribute_vector[6] = 1
    # elif np.float(G.nodes[node_index]['impact_diversity']) < std_min_impact:
    # attribute_vector[6] = 0
    # else:
    attribute_vector[3] = (np.float(G.node[node_index]['impact_diversity']) - mean_impact) / std_impact

    # if np.float(G.nodes[node_index]['scientific_age_diversity']) > std_max_sci:
    # attribute_vector[7] = 1
    # elif np.float(G.nodes[node_index]['scientific_age_diversity']) < std_min_sci:
    # attribute_vector[7] = 0
    # else:
    attribute_vector[4] = (np.float(G.node[node_index]['scientific_age_diversity']) - mean_sci) / std_sci

    return attribute_vector


"""
mean_pooling skip_gram
"""


def mean_pooling(G, skip_gram_vector, walk_length, attritbute_size):
    attribute_vector_total = np.zeros(attritbute_size)
    for index in skip_gram_vector:
        attribute_vector_total += assign_value(G, index)

    return attribute_vector_total / walk_length


"""
Get Neighborhood data
"""


def get_neighborhood_data(G, node_index):
    neighbors = []
    for g in G.neighbors(node_index):
        neighbors.append(np.int(g))

    return neighbors, len(neighbors)


"""
Get neighborhood from new data
"""


def find_neighbor(G_, node_index):
    author_id = G_.nodes[node_index]['author_id']
    author_id2 = G_.nodes[node_index]['author_id2']
    if author_id2 == 0:
        author_id2_v = -1
    else:
        author_id2_v = author_id2
    author_id3 = G_.nodes[node_index]['author_id3']
    if author_id3 == 0:
        author_id3_v = -1
    else:
        author_id3_v = author_id3
    neighbor = [x for x, y in G_.nodes(data=True) if
                y['author_id'] == author_id or y['author_id'] == author_id2_v or y['author_id'] == author_id3_v or
                y['author_id2'] == author_id or y['author_id2'] == author_id2_v or y['author_id2'] == author_id3_v or
                y['author_id3'] == author_id or y['author_id3'] == author_id2_v or y['author_id3'] == author_id3_v]
    size = len(neighbor)
    return neighbor, size


"""
BFS search for nodes
"""


def BFS_search(G, start_node, walk_length):
    walk_ = []
    visited = [start_node]
    BFS_queue = [start_node]
    neighborhood = get_neighborhood_data
    while len(walk_) < walk_length:
        cur = np.int(BFS_queue.pop(0))
        walk_.append(cur)
        cur_nbrs = sorted(get_neighborhood_data(G, cur)[0])
        for node_bfs in cur_nbrs:
            if not node_bfs in visited:
                BFS_queue.append(node_bfs)
                visited.append(node_bfs)
        if len(BFS_queue) == 0:
            visited = [start_node]
            BFS_queue = [start_node]
    return walk_


"""
compute average for one neighborhood node
"""


def average_neighborhood(G, node_index, center_neighbor_size):
    neighbor_vec = assign_value(G, node_index)
    neighbor, neighbor_size = get_neighborhood_data(G, node_index)
    average_factor = 1 / np.sqrt(neighbor_size * center_neighbor_size)

    return neighbor_vec * average_factor


"""
GCN Neighborhood extractor
"""


def GCN_aggregator(G, node_index, attribute_size):
    neighbors, size = get_neighborhood_data(G, node_index)
    aggregate_vector = np.zeros(attribute_size)
    for index in neighbors:
        neighbor_average_vec = average_neighborhood(G, index, size)
        aggregate_vector += neighbor_average_vec

    return aggregate_vector


"""
mean_pooling neighborhood
"""


def mean_pooling_neighbor(G, node_index, attritbute_size):
    neighbors, size = get_neighborhood_data(G, node_index)
    attribute_vector_total = np.zeros(attribute_size)
    for index in neighbors:
        attribute_vector_total += assign_value(G, index)

    return attribute_vector_total / size

"""
Define parameters
"""
batch_size = 20
attribute_size = 5
walk_length = 8
latent_dim = 100
#latent_dim_second = 100
latent_dim_gcn = 8
latent_dim_gcn2 = 100
latent_dim_a = 100
negative_sample_size = 100
data_length = len(list(G.nodes()))

"""
Input of GCN aggregator
"""
x_gcn = tf.placeholder(tf.float32, [None, 1 + walk_length + negative_sample_size, attribute_size])
x_skip = tf.placeholder(tf.float32, [None, walk_length, attribute_size])
x_negative = tf.placeholder(tf.float32, [None, negative_sample_size, attribute_size])
x_label = tf.placeholder(tf.float32, [None, data_length])
"""
Input of center node
"""
x_center = tf.placeholder(tf.float32, [None, attribute_size])
"""
Input of target vector
"""
y_mean_pooling = tf.placeholder(tf.float32, [None, attribute_size])

"""
Input of skip-gram vectors
"""
z_skip_gram = tf.placeholder(tf.float32, [None, walk_length, latent_dim])

z_skip_gram_normalize = tf.math.l2_normalize(z_skip_gram, axis=1)

"""
Input negative sampling samples
"""
z_negative_sampling = tf.placeholder(tf.float32, [None, negative_sample_size, latent_dim])

z_negative_sampling_normalize = tf.math.l2_normalize(z_negative_sampling, axis=1)

"""
Lable for link prediction
"""
Dense_gcn = tf.layers.dense(inputs=x_gcn,
                            units=latent_dim_gcn,
                            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                            activation=tf.nn.relu)

Dense_gcn2 = tf.layers.dense(inputs=x_gcn,
                             units=latent_dim_gcn,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             activation=tf.nn.relu)

Dense_gcn3 = tf.layers.dense(inputs=x_gcn,
                             units=latent_dim_gcn,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             activation=tf.nn.relu)

Dense_gcn4 = tf.layers.dense(inputs=x_gcn,
                             units=latent_dim_gcn,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             activation=tf.nn.relu)

Dense_gcn5 = tf.layers.dense(inputs=x_gcn,
                             units=latent_dim_gcn,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             activation=tf.nn.relu)

Dense_gcn6 = tf.layers.dense(inputs=x_gcn,
                             units=latent_dim_gcn,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             activation=tf.nn.relu)

Dense_gcn7 = tf.layers.dense(inputs=x_gcn,
                             units=latent_dim_gcn,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             activation=tf.nn.relu)

Dense_gcn8 = tf.layers.dense(inputs=x_gcn,
                             units=latent_dim_gcn,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             activation=tf.nn.relu)

Dense_gcn9 = tf.layers.dense(inputs=x_gcn,
                             units=latent_dim_gcn,
                             kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                             activation=tf.nn.relu)

Dense_gcn10 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn11 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn12 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn13 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn14 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn15 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn16 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn17 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn18 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn19 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense_gcn20 = tf.layers.dense(inputs=x_gcn,
                              units=latent_dim_gcn,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

"""
Lable for link prediction
"""
# y_label = tf.placeholder(tf.float32,[None,walk_length+negative_sample_size])

"""
Perform concatenation operation for posterior probability
"""
# concat_posterior = tf.concat([x_center, Dense_gcn, y_mean_pooling],1)

concat_posterior = tf.concat([Dense_gcn, Dense_gcn2, Dense_gcn3,
                              Dense_gcn4, Dense_gcn5, Dense_gcn6,
                              Dense_gcn7, Dense_gcn8, Dense_gcn9,
                              Dense_gcn10, Dense_gcn11, Dense_gcn12,
                              Dense_gcn13, Dense_gcn14, Dense_gcn15,
                              Dense_gcn16, Dense_gcn17, Dense_gcn18,
                              Dense_gcn19, Dense_gcn20], 2)

Dense2_gcn1 = tf.layers.dense(inputs=concat_posterior,
                              units=latent_dim_gcn2,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense2_gcn2 = tf.layers.dense(inputs=concat_posterior,
                              units=latent_dim_gcn2,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense2_gcn3 = tf.layers.dense(inputs=concat_posterior,
                              units=latent_dim_gcn2,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense2_gcn4 = tf.layers.dense(inputs=concat_posterior,
                              units=latent_dim_gcn2,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

Dense2_gcn5 = tf.layers.dense(inputs=concat_posterior,
                              units=latent_dim_gcn2,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.relu)

concat_posterior2 = tf.concat([Dense2_gcn1, Dense2_gcn2, Dense2_gcn3,
                               Dense2_gcn4, Dense2_gcn5], 2)

Dense_layer_fc_gcn = tf.layers.dense(inputs=concat_posterior2,
                                     units=latent_dim,
                                     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                     activation=tf.nn.elu)

"""
Implement sgnn with new structure
"""
idx_origin = tf.constant([0])
idx_skip = tf.constant([i+1 for i in range(walk_length)])
idx_negative = tf.constant([i+1+walk_length for i in range(negative_sample_size)])
x_origin = tf.gather(Dense_layer_fc_gcn,idx_origin,axis=1)
x_skip = tf.gather(Dense_layer_fc_gcn,idx_skip,axis=1)
x_negative = tf.gather(Dense_layer_fc_gcn,idx_negative,axis=1)

#x_origin = tf.gather(x_gcn,idx_origin,axis=1)
#x_skip = tf.gather(x_gcn,idx_skip,axis=1)
#x_negative = tf.gather(x_gcn,idx_negative,axis=1)

x_output = tf.squeeze(x_origin)

doc_regularization = tf.multiply(Dense_layer_fc_gcn,Dense_layer_fc_gcn)

sum_doc_regularization = tf.reduce_sum(tf.reduce_sum(doc_regularization,axis=2),axis=1)

mean_sum_doc = tf.reduce_mean(sum_doc_regularization)

negative_training = tf.broadcast_to(tf.expand_dims(x_negative,1),[batch_size,walk_length,negative_sample_size,latent_dim])

skip_training = tf.broadcast_to(tf.expand_dims(x_skip,2),[batch_size,walk_length,negative_sample_size,latent_dim])

negative_training_norm = tf.math.l2_normalize(negative_training,axis=3)

skip_training_norm = tf.math.l2_normalize(skip_training,axis=3)

dot_prod = tf.multiply(skip_training_norm,negative_training_norm)

dot_prod_sum = tf.reduce_sum(dot_prod,3)

skip_mean = tf.reduce_mean(dot_prod_sum,1)

log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(skip_mean)))

sum_log_dot_prod = tf.reduce_sum(log_dot_prod,1)

positive_training = tf.broadcast_to(x_origin,[batch_size,walk_length,latent_dim])

positive_skip_norm = tf.math.l2_normalize(x_skip,axis=2)

positive_training_norm = tf.math.l2_normalize(positive_training,axis=2)

dot_prod_positive = tf.multiply(positive_skip_norm,positive_training_norm)

dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive,2)

sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive,1)))

negative_sum = tf.math.negative(tf.reduce_sum(tf.math.add(sum_log_dot_prod,sum_log_dot_prod_positive)))

regularized_negative_sum = tf.math.add(negative_sum,mean_sum_doc)

"""
mse loss
"""
Decoding_auto_encoder = tf.layers.dense(inputs=x_origin,
                              units=attribute_size,
                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                              activation=tf.nn.sigmoid)

Decoding_reduce = tf.squeeze(Decoding_auto_encoder)
#mse = tf.losses.mean_squared_error(y_mean_pooling, Decoding_reduce)
mse = tf.losses.mean_squared_error(x_center, Decoding_reduce)

total_loss = tf.math.add(mse,negative_sum)


train_step_auto = tf.train.AdamOptimizer(1e-3).minimize(total_loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

"""
Define get batch 
"""


def get_batch_BFS(G, batch_size, walk_length, length, start_index):
    walk = np.zeros((batch_size, walk_length))
    batch_start_nodes = []
    nodes = np.array(G.nodes())
    for i in range(batch_size):
        walk_single = np.array(BFS_search(G, nodes[i + start_index], walk_length))
        batch_start_nodes.append(nodes[i + start_index])
        walk[i, :] = walk_single
    return walk, batch_start_nodes


"""
get minibatch center data
"""


def get_minibatch(G, index_vector, batch_size, attritbute_size):
    mini_batch = np.zeros((batch_size, attribute_size))
    index = 0
    for node_index in index_vector:
        x_center1 = assign_value(G, node_index)
        mini_batch[index, :] = x_center1
        index += 1

    return mini_batch


"""
get batch neighbor_GCN_aggregate
"""


def get_batch_GCNagg(G, index_vector, batch_size, attribute_size):
    mini_batch_gcn_agg = np.zeros((batch_size, attribute_size))
    index = 0
    for node_index in index_vector:
        single_gcn = GCN_aggregator(G, node_index, attribute_size)
        mini_batch_gcn_agg[index, :] = single_gcn
        index += 1

    return mini_batch_gcn_agg


"""
get batch negative sampling 
"""


def get_batch_negative(G, negative_samples, batch_size, negative_sample_size, attribute_size):
    mini_batch_negative = np.zeros((batch_size, negative_sample_size, attribute_size))
    for i in range(batch_size):
        index = 0
        for node in negative_samples[i, :]:
            negative_sample = assign_value(G, node)
            mini_batch_negative[i, index, :] = negative_sample
            index += 1
    return mini_batch_negative


"""
get batch skip_gram samples
"""


def get_batch_skip_gram(G, skip_gram_vecs, batch_size, walk_length, attribute_size):
    mini_batch_skip_gram = np.zeros((batch_size, walk_length, attribute_size))
    for i in range(batch_size):
        index = 0
        for node in skip_gram_vecs[i, :]:
            skip_gram_sample = assign_value(G, node)
            mini_batch_skip_gram[i, index, :] = skip_gram_sample
            index += 1
    return mini_batch_skip_gram


"""
Uniform sample negative data
"""


def uniform_get_negative_sample(G, skip_gram_vec, center_node, negative_sample_size):
    negative_samples = []
    node_neighbors, neighbor_size = get_neighborhood_data(G, center_node)
    total_negative_samples = 0
    while (total_negative_samples < negative_sample_size):
        index_sample = np.int(np.floor(np.random.uniform(0, np.array(G.nodes()).shape[0], 1)))
        sample = np.int(np.array(G.nodes())[index_sample])
        correct_negative_sample = 1
        for positive_sample in skip_gram_vec:
            if sample == np.int(positive_sample):
                correct_negative_sample = 0
                break
        if correct_negative_sample == 0:
            continue
        for neighborhood_sample in node_neighbors:
            if sample == neighborhood_sample:
                correct_negative_sample = 0
        if sample == center_node:
            correct_negative_sample = 0
        if correct_negative_sample == 1:
            total_negative_samples += 1
            negative_samples.append(sample)

    return negative_samples


"""
get batch mean_pooling
"""


def get_batch_mean_pooling(G, index_vector, batch_size, walk_length, attribute_size):
    mini_batch_mean_pooling = np.zeros((batch_size, attribute_size))
    mini_batch_skip_gram_vectors = np.zeros((batch_size, walk_length))
    index = 0
    for node_index in index_vector:
        # skip_gram_vector = np.array(get_attribute_data_skip_gram(node_index, walk_length))
        skip_gram_vector = np.array(BFS_search(G, node_index, walk_length))
        y_mean_pooling1 = mean_pooling(G, skip_gram_vector, walk_length, attribute_size)
        mini_batch_mean_pooling[index, :] = y_mean_pooling1
        mini_batch_skip_gram_vectors[index, :] = skip_gram_vector
        index += 1

    return mini_batch_mean_pooling, mini_batch_skip_gram_vectors


"""
Convert real number into binary
"""


def convert_binary(y_mean_pool):
    approx = np.floor(y_mean_pool * 100)
    batch_size = y_mean_pool.shape[0]
    batch_b_concat = np.zeros((batch_size, 7 * 8))
    b_concat = []
    for i in range(batch_size):
        for num in approx[0]:
            binary = bin(np.int(num))[2:].zfill(7)
            for j in binary:
                if j == '1':
                    b_concat.append(1)
                if j == '0':
                    b_concat.append(0)
        batch_b_concat[i, :] = b_concat
        b_concat = []

    return batch_b_concat


def get_data(G, batch_size, walk_length, length, start_index_, negative_sample_size):
    # mini_batch_raw = np.array(Node2vec.simulate_walks(batch_size,walk_length))
    mini_batch_raw, start_nodes = get_batch_BFS(G, batch_size, walk_length, length, start_index_)
    negative_samples = np.zeros((batch_size, negative_sample_size))
    negative_samples_vectors = np.zeros((batch_size, negative_sample_size, 8))
    skip_gram_vectors = np.zeros((batch_size, walk_length, 8))
    mini_batch_x = np.zeros((batch_size, 8))
    mini_batch_gcn = np.zeros((batch_size, 8))
    mini_batch_y = np.zeros((batch_size, length))
    mini_batch_x_label = np.zeros((batch_size, length))
    mini_batch_x_y = np.zeros((batch_size, length * 2))
    # mini_batch_x_y = np.zeros((batch_size,2*length))
    mini_batch_x = get_minibatch(G, start_nodes, batch_size)
    batch_GCN_agg = get_batch_GCNagg(G, start_nodes, batch_size)
    mini_batch_y_mean_pool, mini_batch_skip_gram = get_batch_mean_pooling(G, start_nodes, batch_size, walk_length)
    batch_b_concat = convert_binary(mini_batch_y_mean_pool)

    # for i in range(batch_size):
    # negative_samples[i,:] = uniform_get_negative_sample(G,mini_batch_raw[i,:],start_nodes[i],negative_sample_size)

    # negative_samples_vectors = get_batch_negative(G,negative_samples,batch_size,negative_sample_size)

    # skip_gram_vectors = get_batch_skip_gram(G,mini_batch_raw,batch_size,walk_length)

    for i in range(batch_size):
        # index_node = G.nodes[mini_batch_raw[i][0]]['node_index']
        # mini_batch_x[i,:] = 1
        # prob = 1/walk_length
        for j in range(walk_length):
            indexy = G.nodes[mini_batch_raw[i][j]]['index']
            mini_batch_y[i][indexy] = 1 / walk_length  # += prob
        # mini_batch_x_y[i] = np.concatenate((mini_batch_x[i], mini_batch_y[i]),axis=None)
        indexx = G.nodes[start_nodes[i]]['index']
        mini_batch_x_label[i][indexx] = 1
        mini_batch_x_y[i] = np.concatenate((mini_batch_x_label[i], mini_batch_y[i]), axis=None)

    mini_batch_concat_x_y = np.concatenate((mini_batch_x, mini_batch_y_mean_pool), axis=1)

    return mini_batch_x, mini_batch_y, batch_GCN_agg, negative_samples_vectors, skip_gram_vectors, mini_batch_x_label, mini_batch_x_y, mini_batch_y_mean_pool, mini_batch_concat_x_y, batch_b_concat


def get_data_one_batch(G, batch_size, walk_length, length, start_index_, negative_sample_size, attribute_size):
    mini_batch_integral = np.zeros((batch_size, 1 + walk_length + negative_sample_size, attribute_size))
    mini_batch_raw, start_nodes = get_batch_BFS(G, batch_size, walk_length, length, start_index_)
    mini_batch_y = np.zeros((batch_size, length))
    mini_batch_x_label = np.zeros((batch_size, length))
    mini_batch_y_mean_pool = np.zeros(3)

    batch_center_x = get_minibatch(G, start_nodes, batch_size, attribute_size)
    batch_GCN_agg = get_batch_GCNagg(G, start_nodes, batch_size, attribute_size)
    negative_samples = np.zeros((batch_size, negative_sample_size))
    skip_gram_vectors = np.zeros((batch_size, walk_length, attribute_size))
    negative_samples_vectors = np.zeros((batch_size, negative_sample_size, attribute_size))
    for i in range(batch_size):
        mini_batch_integral[i, 0, :] = batch_GCN_agg[i, :]
    """
    mini_batch_y_mean_pool, mini_batch_skip_gram = get_batch_mean_pooling(G, start_nodes, batch_size, walk_length,
                                                                          attribute_size)

    for i in range(batch_size):
        negative_samples[i, :] = uniform_get_negative_sample(G, mini_batch_raw[i, :], start_nodes[i],
                                                             negative_sample_size)

    negative_samples_vectors = get_batch_negative(G, negative_samples, batch_size, negative_sample_size, attribute_size)

    skip_gram_vectors = get_batch_skip_gram(G, mini_batch_raw, batch_size, walk_length, attribute_size)
    for i in range(batch_size):
        mini_batch_integral[i, 1:walk_length + 1, :] = skip_gram_vectors[i, :, :]

    for i in range(batch_size):
        mini_batch_integral[i, walk_length + 1:, :] = negative_samples_vectors[i, :, :]

    
    for i in range(batch_size):
      #index_node = G.nodes[mini_batch_raw[i][0]]['node_index']
      #mini_batch_x[i,:] = 1
      #prob = 1/walk_length
      for j in range(walk_length):
        indexy = G.nodes[mini_batch_raw[i][j]]['node_index']
        mini_batch_y[i][indexy] = 1#/walk_length#+= prob
      #mini_batch_x_y[i] = np.concatenate((mini_batch_x[i], mini_batch_y[i]),axis=None)
      indexx = G.nodes[start_nodes[i]]['node_index']
      mini_batch_x_label[i][indexx] = 1
    """
    for i in range(batch_size):
        indexy = G.node[start_nodes[i]]['node_index']
        mini_batch_y[i][indexy] = 1
        for j in G.neighbors(start_nodes[i]):
            indexy = G.node[j]['node_index']
            mini_batch_y[i][indexy] = 1

    return mini_batch_integral, mini_batch_y, mini_batch_x_label, mini_batch_y_mean_pool, batch_center_x

#for i in range(5):
for j in range(100):
  mini_batch_integral,mini_batch_y,mini_batch_x_label,mini_batch_y_mean_pool,mini_batch_x = get_data_one_batch(G, batch_size,walk_length,data_length,np.int(np.floor(np.random.uniform(0,89498))),negative_sample_size,attribute_size)
  err_ = sess.run([total_loss,train_step_auto],feed_dict={x_gcn:mini_batch_integral,
                                                          y_mean_pooling:mini_batch_y_mean_pool,
                                                          x_center:mini_batch_x})
                                                        #y_label:mini_batch_y,
                                                       # x_label:mini_batch_x_label})
  #print("iteration")
  #print(j)
  print(err_[0])

mini_batch_whole = np.zeros((80000, 1 + walk_length + negative_sample_size, attribute_size))
for i in range(80):
  mini_batch_integral,mini_batch_y,mini_batch_x_label,mini_batch_y_mean_pool,mini_batch_x = get_data_one_batch(G, 1000,walk_length,data_length,i*1000,negative_sample_size,attribute_size)
  mini_batch_whole[i*1000:(i+1)*1000,:,:]=mini_batch_integral
  del mini_batch_integral
  del mini_batch_y
  del mini_batch_x_label
  del mini_batch_y_mean_pool
  del mini_batch_x
  print(i)

mini_batch_part = mini_batch_whole[0:5000,:,:]
embedding_d2v = sess.run([x_output], feed_dict={x_gcn: mini_batch_whole})

embedding_d2v_whole = np.zeros((80000,100))
for i in range(16):
    embedding_d2v = sess.run([x_output],feed_dict={x_gcn:mini_batch_whole[i*5000:(i+1)*5000,:,:]})
    embedding_d2v_whole[i*5000:(i+1)*5000,:]=embedding_d2v[0]

embedding_norm_d2v = np.zeros((80000,100))
score = np.zeros(80000)
embedding_first = embedding_d2v_whole[100]/np.linalg.norm(embedding_d2v_whole,axis=1)[100]
norm = np.linalg.norm(embedding_d2v_whole,axis=1)
for i in range(80000):
  embedding_norm_d2v[i,:] = embedding_d2v_whole[i]/norm[i]
  score[i] =np.matmul(embedding_norm_d2v[i,:].T,embedding_first)
  print(i)
score.min()

from sklearn.cluster import KMeans

embedding_2d_d2v = TSNE(n_components=2).fit_transform(np.array(embedding_d2v_whole))

kmeans = KMeans(n_clusters=4,random_state=0).fit(embedding_norm_d2v)

for i in range(80000):
  if kmeans.labels_[i] == 0:
    color_ = 'green'
  if kmeans.labels_[i] == 1:
    color_ = 'blue'
  if kmeans.labels_[i] == 2:
    color_ = 'red'
  if kmeans.labels_[i] == 3:
    color_ = 'yellow'
  if kmeans.labels_[i] == 4:
    color_ = 'black'
  if kmeans.labels_[i] == 5:
    color_ = 'purple'
  if kmeans.labels_[i] == 6:
    color_ = 'orange'
  if kmeans.labels_[i] == 7:
    color_ = 'pink'

  plt.plot(embedding_2d_d2v[i][0],embedding_2d_d2v[i][1],'.', color = color_,markersize=1)