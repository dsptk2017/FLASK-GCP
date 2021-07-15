#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
#from matplotlib import pylab as plt   
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


df = pd.read_csv('jd_labeled_mannual_team_Label2.csv')




# In[27]:


df=df.sort_values(by=['Label']).reset_index()


# In[28]:


import numpy as np
index=df.Label.index[df.Label.apply(np.isnan)]


# In[29]:


df_label=df[:578]


# In[30]:


df_label.info()
df_unlabel=df['description'][578:]


# import matplotlib.pyplot as plt                                                                    ###
# fig = plt.figure(figsize=(8,6))
# df.Label.count().plot.bar(ylim=0)
# plt.show()

# In[31]:


df_label.Label.value_counts()


# In[32]:


del df_label['index']


# In[33]:


Id_to_Indus={1: 'IT Consutlant software service',
           2:'Retail, manufacturing',
           3: 'Finance, insurance',
           4: 'Federal, department and law',
           5: 'Healthcare pharmaceutical',
           6: 'High Tech'}


# In[34]:


Id_to_Indus.items()


# In[35]:


Indus_to_Id={'IT Consutlant software service': 1,
           'Retail, manufacturing':2,
           'Financial, insurance':3 ,
           'Federal, department and law':4 ,
           'Healthcare pharmaceutical':5 ,
           'High Tech':6 }


# In[36]:


df_label['Label']=df_label['Label'].apply(np.int64)


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=5, norm='l2', ngram_range=(1, 3), stop_words='english')

features = tfidf.fit_transform(df_label['description']).toarray()                      #to array
labels = df_label['Label']
features.shape


# ### self add stop word

# In[38]:


from sklearn.feature_selection import chi2
import numpy as np

N = 10
for Industry, category_id in sorted(Indus_to_Id.items()):
  features_chi2 = chi2(features, labels == category_id)

  indices = np.argsort(features_chi2[0])
    
  feature_names = np.array(tfidf.get_feature_names())[indices]

  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  trigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  #print("# '{}':".format(Industry))
  #print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
  #print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
  #print("  . Most correlated trigrams:\n       . {}".format('\n       . '.join(trigrams[-N:])))


# In[39]:


#################################################bench mark MultinomialNB #################################

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[40]:


X=df_label['description']
y=df_label['Label']


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)


# In[42]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer          #
from nltk.corpus import stopwords  
from nltk.stem.snowball import EnglishStemmer
import re


# In[44]:


def stem_tokenizer(text):
    stemmer = EnglishStemmer(ignore_stopwords=True)
    words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
    words = [stemmer.stem(word) for word in words]
    return words 


# In[45]:


tfidf = TfidfVectorizer(stop_words=stopwords.words('english'),
                        tokenizer=stem_tokenizer,
                        lowercase=True,
                        max_df=0.5,
                        min_df=5,
                        ngram_range=(1, 3)
                       )


# In[46]:


import nltk
StopWords = nltk.corpus.stopwords.words('english')

#print(len(StopWords))
#print(StopWords)


# 4  federal,    6  high tech

# In[47]:



newStopWords = ['fortune 500','years required','specialized experience','years required','dreams','succeeding succeeding',
               'realize dreams','letâ','fortune 500','ignite','cities','letâ world','world forward','motion big',
               'moving 600','help drivers','seek opportunity','ignite opportunity','uber ignite','world motion','problems help']


StopWords.extend(newStopWords)


# In[48]:


##############################################################################################


# In[49]:


lgclassifier = Pipeline([('tfidf', tfidf), ('lg', LogisticRegression(class_weight="balanced"))])


# In[40]:


lgclassifier = lgclassifier.fit(X_train, y_train)


# In[ ]:


import pickle
with open('lgclassifier.pkl','wb') as f:
    pickle.dump(lgclassifier,f)

