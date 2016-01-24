import bcubed
import numpy
import random
from os import listdir
from os.path import join
"""
import warnings
warnings.filterwarnings("ignore")
"""
eps = 0.001
words_list_path = "D:\\OneDrive\\RPI\\Acads\\1_1\\NLP_Heng_Ji_6962\\7\\Assignment3\\triggerdata"
word_dict = {}
master_word_list = []
dict_gold = {}
dict_value = 0
for f in listdir(words_list_path):
    f_obj = open(join(words_list_path,f))
    
    for line in f_obj:
        lsl = line.strip().lower()
        dict_gold[lsl] = dict_gold.get(lsl,[])
        dict_gold[lsl].append(dict_value)
        master_word_list.append(lsl)
    f_obj.close()
    dict_value+=1
    
for key in dict_gold.keys():
    dict_gold[key] = set(dict_gold[key])
    
    



num_clusters = 35
word_vec_dict = {}
word_vec_path = "D:\\Zmisc\\zipped\\glove.6B.50d.txt"


for line in open(word_vec_path):
    li = line.split()
    word_vec_dict[li[0].lower()] = numpy.asarray(li[1:])
    
len_vec = len(word_vec_dict['aqm'])

random.shuffle(master_word_list)

centroids = (numpy.random.rand(num_clusters,len_vec)-0.5)*2
#print centroids

    
clusters = [0]
clusters*= len(master_word_list)
count = 0

print "Master word list length" , len(master_word_list)

for i in range(len(master_word_list)):
   word = master_word_list[i]
   word_vec_dict[word] = word_vec_dict.get(word,{}) 


while True:
    k = 0
    sums = numpy.zeros((num_clusters,len_vec))
    counters = numpy.zeros(num_clusters)    
    clusters_prev = numpy.copy(clusters)
    
    
    for i in range(len(master_word_list)):
        word = master_word_list[i]
        if len(word_vec_dict[word])==0:
            continue
        k+=1
        word_vec_dict[word] = numpy.asarray(word_vec_dict[word],dtype=float)
        closest_cosine = -1
        for j in range(len(centroids)):
            
            closeness = numpy.dot(word_vec_dict[word],centroids[j,:])
            if closeness > closest_cosine:
                clusters[i] = j
                closest_cosine = closeness
    
    
    for i in range(len(master_word_list)):
        word = master_word_list[i]
        if len(word_vec_dict[word])==0:
            continue
        sums[clusters[i],:] = numpy.sum([sums[clusters[i],:], word_vec_dict[word]],axis=0)
        
        counters[clusters[i]] += 1 
    
    for i in range(num_clusters):
        centroids[i,:] = numpy.divide(sums[i,:],counters[i])
        #print sums[i,:]
        #print counters[i]
    
    #print centroids
    #print count
    count+=1
    
    if abs(numpy.linalg.norm(clusters-clusters_prev)) < eps:
        break

final_clust_lis = list()    
for x in range(num_clusters):
    final_clust_lis.append(list())


dict_pred ={}

    
for i in range(len(master_word_list)):
    word = master_word_list[i]
    #if len(word_vec_dict[word])==0:
        #continue
    final_clust_lis[clusters[i]].append(word)
    dict_pred[word] = dict_pred.get(word,[])
    dict_pred[word].append(clusters[i])
    

for key in dict_pred.keys():
    dict_pred[key] = set(dict_pred[key])
    
precision = bcubed.precision(dict_pred, dict_gold)
recall = bcubed.recall(dict_pred, dict_gold)


print "Precision = ", precision
print "Recall = ", recall
print "F-Score = ", (2*precision*recall)/(precision+recall)
    
#print centroids  
#print count  
#print counters
