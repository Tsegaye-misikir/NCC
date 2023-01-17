# -*- coding: utf-8 -*-
"""

Part of: Thesis Project - Learning interlingual document representations 
(Marc Lenz, 2021)
---

This File contains helper function used for the Thesis. 

"""

from collections import defaultdict
from gensim import corpora, models
import csv
import itertools 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import linear_model
import tensorflow as tf
import pickle

def mate_retrieval_score(l1_vecs, l2_vecs):
    sim_matrix = cosine_similarity(l1_vecs, l2_vecs)
    mate_scores = np.diagonal(sim_matrix)
    
    ranks = []
    for k in range(len(mate_scores)):
        mate_score = mate_scores[k]
        mate_rank = np.sum(sim_matrix[k] >= mate_score)
        ranks.append(mate_rank)
    mates_retrieved = ranks.count(1)
    retrieval_score = mates_retrieved/len(mate_scores)
    return retrieval_score


class LCA_Model:
    def __init__(self, l1_docs, l2_docs, l1_lsi_model, l2_lsi_model, clf_type = "nn"):
        self.l1_lsi_model = l1_lsi_model
        self.l2_lsi_model = l2_lsi_model
        self.clf_type = clf_type
        self.train_model(l1_docs, l2_docs, l1_lsi_model, l2_lsi_model)
        
    def train_model(self, l1_docs, l2_docs, l1_lsi_model, l2_lsi_model):
        l1_lsi_vecs = l1_lsi_model.create_embeddings(l1_docs)
        l2_lsi_vecs = l2_lsi_model.create_embeddings(l2_docs)
        #l1_to_l2_clf = linear_model.SGDRegressor(penalty = "l2")
        #l2_to_l1_clf = linear_model.SGDRegressor(penalty = "l2")
        
        if self.clf_type == "linear":
            l1_to_l2_clf = linear_model.LinearRegression()
            l2_to_l1_clf = linear_model.LinearRegression()
            l1_to_l2_clf.fit(l1_lsi_vecs, l2_lsi_vecs)
            l2_to_l1_clf.fit(l2_lsi_vecs, l1_lsi_vecs)      
            self.l1_to_l2_clf = l1_to_l2_clf
            self.l2_to_l1_clf = l2_to_l1_clf
            
        if self.clf_type == "nn":
            #Parameters for NN
            loss_function = "MSE" #tf.keras.losses.CosineSimilarity() #"MSE"
            optimizer_f = "adam"
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            dim = self.l1_lsi_model.dimension
            
            #First classifier         
            l1_to_l2_clf = tf.keras.Sequential([
                tf.keras.layers.Dense(dim, activation= None)]) # kernel_regularizer='l2'
            
            l1_to_l2_clf.compile(optimizer = optimizer_f,
                          loss = loss_function,#tf.keras.losses.CosineSimilarity(),
                          metrics=['MSE'])

            l1_to_l2_clf.fit(np.asarray(l1_lsi_vecs), np.asarray(l2_lsi_vecs), 
                             epochs=200, 
                             validation_data = (np.asarray(l1_lsi_vecs), np.asarray(l2_lsi_vecs)),
                             #validation_split=0.2,
                             callbacks = [callback])
            
            #Second classifier
            l2_to_l1_clf = tf.keras.Sequential([
                tf.keras.layers.Dense(dim, activation= None)]) #, kernel_regularizer='l2'
            
            l2_to_l1_clf.compile(optimizer = optimizer_f,
                          loss= loss_function, #tf.keras.losses.CosineSimilarity(),
                          metrics=['MSE'])

            l2_to_l1_clf.fit(np.asarray(l2_lsi_vecs), np.asarray(l1_lsi_vecs), 
                             epochs=200,  
                             validation_data = (np.asarray(l1_lsi_vecs), np.asarray(l2_lsi_vecs)),
                             #validation_split=0.2,
                             callbacks = [callback])          
            self.l1_to_l2_clf = l1_to_l2_clf
            self.l2_to_l1_clf = l2_to_l1_clf      
            
    def create_embeddings(self, docs, language="l1"):

        if language == "l1":
            l1_embeddings = self.l1_lsi_model.create_embeddings(docs)
            if self.clf_type == "nn":
                l2_embeddings = self.l1_to_l2_clf.predict(np.asarray(l1_embeddings))
            elif self.clf_type == "linear":
                l2_embeddings = self.l1_to_l2_clf.predict(l1_embeddings)
        if language == "l2":
            l2_embeddings = self.l2_lsi_model.create_embeddings(docs)
            if self.clf_type == "nn":
                l1_embeddings = self.l1_to_l2_clf.predict(np.asarray(l2_embeddings))
            elif self.clf_type == "linear":
                l1_embeddings = self.l2_to_l1_clf.predict(l2_embeddings)
        
        l1_embeddings = np.asarray(l1_embeddings)
        l2_embeddings = np.asarray(l2_embeddings)

        return list(np.concatenate((l1_embeddings, l2_embeddings), axis=1)) 
    
    
    
     


class Vector_Lsi_Model:
    def __init__(self, docs, dimension=50):
        self.docs = docs
        self.dimension = dimension
        self.train_model(docs, dimension)
        
    def train_model(self, docs, dimension):
        dictionary, corpus = create_corpus(docs)
        tfidf_model = models.TfidfModel(corpus)
        corpus_tfidf = tfidf_model[corpus]
        lsi_model = models.LsiModel(corpus_tfidf, 
                            id2word=dictionary, 
                            num_topics=dimension
                            #onepass True/False determines which algo is used
                            #onepass = False,
                            #power_iters = 1,
                            #extra_samples= 100
                            )  
        self.dictionary = dictionary
        self.tfidf_model = tfidf_model
        self.lsi_model = lsi_model
    
    def create_embeddings(self, docs):
        vecs = []
        for doc in docs:
            vec_bow = self.dictionary.doc2bow(doc)
            bow_tfidf = self.tfidf_model[vec_bow]
            vec_lsi = self.lsi_model[bow_tfidf]  
            vec_rep = np.asarray(list(zip(*vec_lsi))[1])
            #This is here because of weird error, has to be fixed
            if vec_rep.shape[0]!= self.dimension:
                 vecs.append(np.zeros(self.dimension))
            else:
                vecs.append(vec_rep)
        return vecs
    
    def save(self, filename):
        outfile = open(filename,'wb')
        pickle.dump(self, outfile)
        outfile.close()

    
    
def train_lsi_model(french_docs, english_docs, dimension, sample_size):
     
    if sample_size <= len(english_docs):
        french_docs = french_docs[:sample_size]
        english_docs = english_docs[:sample_size]
        
    french_dictionary, french_corpus = create_corpus(french_docs)
    english_dictionary, english_corpus = create_corpus(english_docs)

    french_tfidf = models.TfidfModel(french_corpus)
    french_corpus_tfidf = french_tfidf[french_corpus]

    english_tfidf = models.TfidfModel(english_corpus)
    english_corpus_tfidf = english_tfidf[english_corpus]

    french_lsi_model = models.LsiModel(french_corpus_tfidf, 
                            id2word=french_dictionary, 
                            num_topics=dimension)  

    english_lsi_model = models.LsiModel(english_corpus_tfidf, 
                            id2word=english_dictionary, 
                            num_topics=dimension) 

    lsi_models = {"en": english_lsi_model,
                  "en_tfidf": english_tfidf,
                  "en_dict": english_dictionary,
                  "fr": french_lsi_model,
                  "fr_tfidf": french_corpus_tfidf,
                  "fr_dict": french_dictionary,}

    return lsi_models


def extract_docs_from_jrq_xml(tree, 
                              src_language = "english",
                              trg_language = "french",
                              align="sections",
                              doc_limit = 1000):
    """
    
    Takes an XML element tree of JRQ-Arquis Corpus and creates a list of 
    aligned paragraphs. 
    
    Use like that: 
        1. get aligned JRQ-Arquis corpus data (XML)
        2. load via import xml.etree.ElementTree as ET, tree = ET.parse(<file>)
        3. use this function on tree
    
    parameters - align = "sections" or "documents" describes if you want to obtain
    a list of aligned sections or aligned documents
    language

    """
    root = tree.getroot()
    
    documents = []
    doc_count = 0
    language_keys = {"s1": src_language, "s2": trg_language}
    
    #Get aligned sections
    if align == "sections":
        for elem in root.iter("linkGrp"):
            sections = []
            for link in elem.iter("link"):
                #Get aligned sections
    
                    section = dict()      
                    for e in link.iter():
                        if e.tag in language_keys.keys():
                            language_key = language_keys[e.tag]
                            section_content = e.text
                            section[language_key] = section_content
                    sections.append(section)
            if len(sections) > 0:
                    documents.append(sections)
                    doc_count += 1
                    if doc_count >= doc_limit:
                        return documents
                
    #Get aligned documents
    if align == "documents":
        for elem in root.iter("linkGrp"):
            doc = {src_language: "", trg_language: ""}
            for link in elem.iter("link"):
                #Get aligned sections   
                    for e in link.iter():
                        if e.tag in language_keys.keys():
                            language_key = language_keys[e.tag]
                            section_content = e.text
                            if section_content != None:
                                doc[language_key] = doc[language_key] + str(section_content)             
            if len(doc) > 0:
                    documents.append(doc)
                    doc_count += 1
                    if doc_count >= doc_limit:
                        return documents
                

    return documents

def dict_combinations(obj):
  keys, values = zip(*obj.items())
  permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
  return permutations_dicts


def filter_docs(l1_docs, l2_docs, min_len=1, max_len=10000):
    filtered_l1_docs = []
    filtered_l2_docs = []
    for k in range(len(l1_docs)):
        l1_doc = l1_docs[k]
        l2_doc = l2_docs[k]
        word_num_l1_doc = len(l1_doc)
        word_num_l2_doc = len(l2_doc)
        min_condition = word_num_l1_doc >= min_len and word_num_l2_doc >= min_len
        max_condition = word_num_l1_doc <= max_len and word_num_l2_doc <= max_len 
        if min_condition and max_condition:
            filtered_l1_docs.append(l1_doc)
            filtered_l2_docs.append(l2_doc)
    return filtered_l1_docs, filtered_l2_docs


def save_docs(l1_docs, l2_docs, l1_destination, l2_destination):
    # open a file, where you ant to store the data
    with open(l1_destination, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for doc in l1_docs:
            spamwriter.writerow(doc)
    
    with open(l2_destination, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for doc in l2_docs:
            spamwriter.writerow(doc)

def read_docs(file_name1, file_name2):
    fd = []
    with open(file_name1, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            fd.append(row)
    
    ed = []
    with open(file_name2, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            ed.append(row)
    return fd, ed


def create_corpus(texts, filter_extremes=True):
    """
    
    Function to create a corpus file out of a list of texts to prepare
    data for model training using Gensim

    """
    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]
    
    dictionary = corpora.Dictionary(texts)
    if filter_extremes == True:
        dictionary.filter_extremes(no_below=5, no_above=0.8)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    return dictionary, corpus
