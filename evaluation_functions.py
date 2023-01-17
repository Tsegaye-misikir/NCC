import numpy as np
import scipy as sp
from tqdm import tqdm

from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from functools import partial
from itertools import combinations
from gensim import corpora, models, matutils

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from gensim import models
import itertools
from thesis_code.Utils import create_corpus, read_docs, mate_retrieval_score

import pickle

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import regularizers


def mate_retrieval(l1_vecs, l2_vecs):
    sim = np.dot(normalize(l1_vecs),normalize(l2_vecs).T)
    '''Mate retrieval rate - the rate when the most symmetric document is ones translation.'''
    return sum([sim[i].argmax()==i for i in range(sim.shape[0])])/sim.shape[0]

def rank(val,a):
    if sp.sparse.issparse(a):
        return a[a>=val].shape[1] #if a is sparse
    return len(a[a>=val]) # if a is dense

def reciprocal_rank_hub(l1_vecs, l2_vecs):
    '''Mean reciprocal rank'''
    sim = cosine_similarity(l1_vecs, l2_vecs)

    l2_sims = cosine_similarity(l2_vecs,l2_vecs)
    sorted_l2 = np.sort(l2_sims)[::-1]
    hub_weighting_l2 = np.mean(sorted_l2[:, 2:200], axis=1)

    l1_sims = cosine_similarity(l1_vecs,l1_vecs)
    sorted_l1 = np.sort(l1_sims)[::-1]
    hub_weighting_l1 = np.mean(sorted_l1[:, :200], axis=1)
    
    sim1 = sim - hub_weighting_l1
    sim2 = sim - hub_weighting_l2

    sim = sim1 + sim2.T

    return sum([1/rank(sim[i,i],sim[i]) for i in range(sim.shape[0])])/sim.shape[0]


def reciprocal_rank(l1_vecs, l2_vecs):
    '''Mean reciprocal rank'''
    sim = cosine_similarity(l1_vecs, l2_vecs)

    return sum([1/rank(sim[i,i],sim[i]) for i in range(sim.shape[0])])/sim.shape[0]

def comp_scores(l1_vecs, l2_vecs):
  return [mate_retrieval(l1_vecs, l2_vecs),reciprocal_rank(l1_vecs, l2_vecs),  reciprocal_rank_hub(l1_vecs, l2_vecs)]


def evaluate_baseline_lca_model_ort(X_train_in, X_test_in, Y_train_in, Y_test_in, dimensions, evaluation_function):
  """
  Input: x_train1, x_test1, x_train2, x_test2 | np.darray - matrices, 
         dimensions | list of integers, dimensons which should be tested
         evaluation function | function which evalutates the results

  The columns of each matrix are document vectors. 
  Language 1 (source) : x_train1, x_test1 | Language 2 (target) : x_train2, x_test2

  Output: Results of the evaluation function for each dimension. 

  """
  scores = []

  for dimension in dimensions:
      #Transpose matrices and reduce to given dimensions
      X_train, X_test = X_train_in[: ,:dimension].T,  X_test_in[:,:dimension].T
      Y_train, Y_test = Y_train_in[:,:dimension].T, Y_test_in[:,:dimension].T
      #Compute SVD to obtain best orthogonal mapping by UV.T 
      u, s, vh = np.linalg.svd(np.dot(X_train, Y_train.T))
      B = np.dot(u, vh)
      #Define the obtained mapping and apply it to the test-data
      linear_mapping = lambda x: np.dot(x, B)
      score = evaluation_function(Y_test.T, linear_mapping(X_test.T))
      scores.append(score)
  return scores 

def evaluate_baseline_lca_model(X_train_in, X_test_in, Y_train_in, Y_test_in, dimensions, evaluation_function):

  """
  Input: x_train1, x_test1, x_train2, x_test2 | np.darray - matrices, 
         dimensions | list of integers, dimensons which should be tested
         evaluation function | function which evalutates the results

  The columns of each matrix are document vectors. 
  Language 1 (source) : x_train1, x_test1 | Language 2 (target) : x_train2, x_test2

  Output: Results of the evaluation function for each dimension. 

  """
  
  
  scores = []
  for dimension in dimensions:
      #Transpose matrices and reduce to given dimensions
      X_train, X_test = X_train_in[: ,:dimension].T,  X_test_in[:,:dimension].T
      Y_train, Y_test = Y_train_in[:,:dimension].T, Y_test_in[:,:dimension].T

      B = np.dot(np.linalg.pinv(Y_train.T), X_train.T)
      linear_mapping = lambda x: np.dot( B.T, x)

      score = evaluation_function(X_test.T, linear_mapping(Y_test).T)
      scores.append(score)
  return scores 


def plot_parameter_graph(dimensions, scores, title, xlabel = "Dimensions", ylabel = "Reciprocal Rank", pair_list=None):
  figure(figsize=(9, 6))

  for k, score in enumerate(scores):
    if pair_list == None:
        plt.plot(dimensions, scores[k], alpha=0.8, label="Language Pair: {}".format(k))
    else:
        plt.plot(dimensions, scores[k], alpha=0.8, label="Language Pair: {} -> {}".format(pair_list[k][0], pair_list[k][1]))
  avg = np.mean(np.asarray(scores), axis=0)
  plt.plot(dimensions, avg, c="r", label="Average Score",linewidth=3.0)

  max_ind = np.argmax(avg)

  plt.scatter(dimensions[max_ind], avg[max_ind], c="k")
  plt.text(dimensions[max_ind]+0.005, avg[max_ind]+0.005, 
          "Dimension: {} \nMean Score: {}".format(dimensions[max_ind],str(avg[max_ind])[:4] ),
          fontsize= 12
              )
  plt.title(title, fontsize=13)
  plt.xlabel("Dimensions")
  plt.ylabel("Reciprocal Rank")
  plt.ylim(0.85,1)
  plt.legend()
  plt.show()

def evaluate_lcc_model( en_train_matrix,
                en_test_matrix,
                fr_train_matrix,
                fr_test_matrix,
                dimensions, 
                evaluation_function):

    scores = []


    for dimension in tqdm(dimensions):
        en = en_train_matrix[: ,:dimension] - np.mean(en_train_matrix[:,:dimension], axis=0)
        fr = fr_train_matrix[: ,:dimension] - np.mean(fr_train_matrix[:,:dimension], axis=0)
        sample_size = en.shape[0]
        zero_matrix = np.zeros((sample_size, dimension))
        X1 = np.concatenate((en, zero_matrix), axis = 1)
        X2 = np.concatenate((zero_matrix, fr), axis= 1)
        X = np.concatenate((X1, X2), axis = 0)
        Y1 = np.concatenate((en, fr), axis = 1)
        Y2 = np.concatenate((en, fr), axis = 1)
        Y = np.concatenate((Y1, Y2), axis = 0)

        reg = linear_model.RidgeCV(alphas=[1e-10, 1e-3, 1e-2, 1e-1, 1, 10])
        reg.fit(X,Y)
        pca = PCA(n_components= int(dimension))
        pca.fit(reg.predict(X))
        rrr = lambda X: np.matmul(pca.transform(reg.predict(X)), pca.components_)

        #sample_size = len(en_docs_test)
        en = en_test_matrix[: ,:dimension] - np.mean(en_train_matrix[:,:dimension], axis=0)
        fr = fr_test_matrix[: ,:dimension] - np.mean(fr_train_matrix[:,:dimension], axis=0)
        zero_matrix = np.zeros((en_test_matrix.shape[0], dimension))
        X1 = np.concatenate((en, zero_matrix), axis = 1)
        X2 = np.concatenate((zero_matrix, fr), axis= 1)
        X = np.concatenate((X1, X2), axis = 0)
        english_encodings_lcc = rrr(X1)
        french_encodings_lcc = rrr(X2)
        score = evaluation_function(english_encodings_lcc, french_encodings_lcc)
        scores.append(score)

    return scores





def evaluate_cllsi(fr_docs_train, fr_docs_test, en_docs_train, en_docs_test, dimensions, evaluation_function):


  scores = []

  dictionary, l1_corpus = create_corpus(fr_docs_train)
  d1 = len(dictionary)
  dictionary.add_documents(en_docs_train)


  multiling_docs = []

  for k in range(len(fr_docs_train)):
      fr_doc = fr_docs_train[k]
      en_doc = en_docs_train[k]
      multiling_docs.append(fr_doc+en_doc)


  bow_reps = [dictionary.doc2bow(line) for line in multiling_docs]

  multiling_tfidf = models.TfidfModel(bow_reps)
  multiling_corpus_tfidf = multiling_tfidf[bow_reps]

  for dimension in dimensions:
      multi_lsi_model = models.LsiModel(multiling_corpus_tfidf, 
                              id2word=dictionary, 
                              num_topics=dimension)  
          
      french_vecs = []
      for doc in fr_docs_test:
          vec_bow = dictionary.doc2bow(doc)
          bow_tfidf = multiling_tfidf[vec_bow]
          vec_lsi = multi_lsi_model[bow_tfidf]  
          vec_rep = np.asarray(list(zip(*vec_lsi))[1])
          if vec_rep.shape[0]!= dimension:
            french_vecs.append(np.zeros(dimension))
          else:
            french_vecs.append(vec_rep)

      english_vecs = []
      for doc in en_docs_test:
          vec_bow = dictionary.doc2bow(doc)
          bow_tfidf = multiling_tfidf[vec_bow]
          vec_lsi = multi_lsi_model[bow_tfidf]  
          vec_rep = np.asarray(list(zip(*vec_lsi))[1])
          #This is here because of weird error, has to be fixed
          if vec_rep.shape[0]!= dimension:
              english_vecs.append(np.zeros(dimension))
          else:
              english_vecs.append(vec_rep)


      score = evaluation_function(english_vecs, french_vecs)

      scores.append(score)
  return scores


def tfidf(data,no_below=1,no_above=0.8):
    '''
    Input: (train,test) list of list of tokens
    Output: (x_train,x_test) tfidf matrixes    
    '''
    train, test = data
    dict = corpora.Dictionary(train)
    dict.filter_extremes(no_below=no_below,no_above=no_above)
    train_bow = [dict.doc2bow(doc) for doc in train]
    test_bow  = [dict.doc2bow(doc) for doc in test]
    tfidf =  models.TfidfModel(train_bow+test_bow)
    x_train = tfidf[train_bow]
    x_test = tfidf[test_bow]
    return (x_train, x_test)

def evaluate_improved_cllsi(x_train1_in,x_test1_in,x_train2_in,x_test2_in , dimensions, evaluation_function):
    scores = []

    for k in dimensions:
      x_train1,x_test1 = tfidf(data = (x_train1_in,x_test1_in))
      x_train2,x_test2 = tfidf(data = (x_train2_in,x_test2_in))

      n_train, n_test = len(x_train1), len(x_test1)

    
      X1= matutils.corpus2csc(list(x_train1) +list( x_test1))
      X2= matutils.corpus2csc(list(x_train2) + list(x_test2))

      x_train1,x_train2 = X1[:,:n_train], X2[:, :n_train]
      x_test1,x_test2 = X1[:,n_train:], X2[:,n_train:]
      

      x = sp.sparse.vstack([x_train1,x_train2])
      x = matutils.Sparse2Corpus(x)

      lsa = models.LsiModel(x,num_topics=k)
      n = x_train1.shape[0]
      U = lsa.projection.u
      U1, U2 = U[:n,:], U[n:,:]
      p1,p2 = sp.sparse.csr_matrix(np.linalg.pinv(U1)), sp.sparse.csr_matrix(np.linalg.pinv(U2))  
      a1,a2 = np.dot(x_test1.T,p1.T).todense(), np.dot(x_test2.T,p2.T).todense()

      score = evaluation_function(a1,a2)
      scores.append(score)
    return scores




def evaluate_nnca(l1_train, l1_test, l2_train, l2_test, 
                  dimensions,
                  evaluation_function = reciprocal_rank,
                  neurons = [100],
                  activation_function = "relu",
                  max_epochs = 100,
                  dropout = 0.2,
                  optimizer = "adam",
                  loss = "MSE"
                             ):
  callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
  scores = []


  for idz, dimension in enumerate (dimensions):

    if len(neurons) == 1:
      x = tf.keras.layers.Input(shape=(dimension,))
      d1 = tf.keras.layers.Dropout(dropout)(x)
      emb = tf.keras.layers.Dense(neurons[0], activation = activation_function)(d1)
      d2 = tf.keras.layers.Dropout(dropout)(emb)
      y = tf.keras.layers.Dense(dimension, activation = activation_function)(d2)
      model = Model(x, y)
    elif len(neurons) == 3:
      x = tf.keras.layers.Input(shape=(dimension,))
      d1 = tf.keras.layers.Dropout(dropout)(x)
      h1 = tf.keras.layers.Dense(neurons[0], activation = activation_function)(d1)
      emb = tf.keras.layers.Dense(neurons[1], activation = activation_function)(h1)
      h2 = tf.keras.layers.Dense(neurons[2], activation = activation_function)(emb)
      d2 = tf.keras.layers.Dropout(dropout)(h2)
      y = tf.keras.layers.Dense(dimension, activation=None)(d2)
      model = Model(x, y)

    if loss == "cosine_sim":
      loss = tf.keras.losses.CosineSimilarity()

    l1_to_l2_clf = Model(x, y)
    l2_to_l1_clf = Model(x, y)

    l1_to_l2_clf.compile(optimizer = optimizer,
            loss = loss,
            metrics= tf.keras.losses.CosineSimilarity()
            )

    l2_to_l1_clf.compile(optimizer = optimizer,
            loss = loss,
            metrics= tf.keras.losses.CosineSimilarity())


    hist1 = l1_to_l2_clf.fit(l1_train[: ,:dimension], 
                      l2_train[:,:dimension], 
                      epochs=max_epochs, 
                      validation_data = (l1_test[: ,:dimension], l2_test[: ,:dimension]), 
                      callbacks=[callback])

    hist2 = l2_to_l1_clf.fit(l2_train[: ,:dimension], 
                      l1_train[: ,:dimension],
                      epochs=max_epochs, 
                      validation_data = (l2_test[: ,:dimension], l1_test[: ,:dimension]), 
                      callbacks=[callback])  

    fake_fr = l1_to_l2_clf.predict(l1_test[: ,:dimension])
    fake_en = l2_to_l1_clf.predict(l2_test[: ,:dimension])

    merged_trans_vecs = np.concatenate((fake_en, l2_test[:,:dimension]), axis = 1)
    real_vecs = np.concatenate((l1_test[:,:dimension], fake_fr), axis = 1)


    score = evaluation_function(merged_trans_vecs, real_vecs)
    scores.append(score)
  return scores, hist1, hist2


import tensorflow as tf
from tensorflow.keras import Model

def evaluate_nncc(l1_train, l1_test, l2_train, l2_test, 
                             dimensions ,
                             evaluation_function = reciprocal_rank,
                             neurons = [100],
                             activation_function = "relu",
                             max_epochs = 100,
                             dropout = 0.2,
                             optimizer = "adam",
                             loss = "MSE") :

  callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
  l1_train, l1_test, l2_train, l2_test = np.asarray(l1_train), np.asarray(l1_test), np.asarray(l2_train), np.asarray(l2_test)


  scores = []


  for idz, dimension in enumerate (dimensions):
      en = l1_train[:,:dimension] - np.mean(l1_train[:,:dimension], axis=0)
      fr = l2_train[:,:dimension] - np.mean(l2_train[:,:dimension], axis=0)
      zero_matrix = np.zeros((en.shape[0], dimension))
      X1 = np.concatenate((en, zero_matrix), axis = 1)
      X2 = np.concatenate((zero_matrix, fr), axis= 1)
      X = np.concatenate((X1, X2), axis = 0)
      Y1 = np.concatenate((en, fr), axis = 1)
      Y2 = np.concatenate((en, fr), axis = 1)
      Y = np.concatenate((Y1, Y2), axis = 0)

      if len(neurons) == 1:
        x = tf.keras.layers.Input(shape=(dimension*2,))
        d1 = tf.keras.layers.Dropout(dropout)(x)
        emb = tf.keras.layers.Dense(neurons[0], activation = activation_function)(d1)
        d2 = tf.keras.layers.Dropout(dropout)(emb)
        y = tf.keras.layers.Dense(dimension*2, activation = activation_function)(d2)
        model = Model(x, y)
      elif len(neurons) == 3:
        x = tf.keras.layers.Input(shape=(dimension*2,))
        d1 = tf.keras.layers.Dropout(dropout)(x)
        h1 = tf.keras.layers.Dense(neurons[0], activation = activation_function)(d1)
        emb = tf.keras.layers.Dense(neurons[1], activation = activation_function)(h1)
        h2 = tf.keras.layers.Dense(neurons[2], activation = activation_function)(emb)
        d2 = tf.keras.layers.Dropout(dropout)(h2)
        y = tf.keras.layers.Dense(dimension*2, activation=None)(d2)
        model = Model(x, y)

      if loss == "cosine_sim":
        loss = tf.keras.losses.CosineSimilarity()
      model.compile(optimizer = optimizer,
                    loss =  loss, #"MSE",#tf.keras.losses.CosineSimilarity(),
                    metrics=[tf.keras.losses.CosineSimilarity()])

      history = model.fit(X, Y, epochs=max_epochs, validation_split=0.1, callbacks=[callback])



      en = l1_test[: ,:dimension] - np.mean(l1_train[:,:dimension], axis=0)
      fr = l2_test[: ,:dimension] - np.mean(l1_train[:,:dimension], axis=0)
      zero_matrix = np.zeros((l1_test.shape[0], dimension))

      X1 = np.concatenate((en, zero_matrix), axis = 1)
      X2 = np.concatenate((zero_matrix, fr), axis= 1)
      X = np.concatenate((X1, X2), axis = 0)
      
      encoder = Model(x, emb)
      english_encodings_nncc = encoder.predict(X1)
      french_encodings_nncc = encoder.predict(X2)

      score = evaluation_function(english_encodings_nncc, french_encodings_nncc)


      scores.append(score)

  return scores, history