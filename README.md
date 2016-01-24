# K-Means-Clusters-Words
An implementation of the k-Means clustering algorithm in Python to group words into clusters using word-to-vector representations.

The objective of this Python project was to cluster related words using the k-Means algorithm. 
The target words were present as individual tokens in 35 files (for 35 clusters). 
These words were converted into vectors using the pre-trained vectors of Glove. 
Link: http://nlp.stanford.edu/projects/glove/

Download link for pre-trained vectors: http://nlp.stanford.edu/data/glove.6B.zip. 

The file trained on a corpus of 6 billion words with each vector’s length 50 has been used. 
These vectors were clustered by to the K-means algorithm. The clustering has been evaluated using the B-cubed measure.

The Results are as follows:

Precision =  0.453965927911

Recall =  0.358473203728

F-Score =  0.400607539012

Error Analysis: There are two main reasons of the low F-score. 

1)	The k-means clustering algorithm usually creates exclusive clusters, meaning one word could not be in both clusters.
However, in the ground truth, many words are in multiple clusters. For example, the acronym ‘aka’ is there in 3 clusters. 

2)	There are many multi-word phrases among the target words. 
For example, age of, died at, dies at. The pre-trained vectors from Glove only have single words. 
In fact, more than half of the words/phrases are absent in Glove. 

