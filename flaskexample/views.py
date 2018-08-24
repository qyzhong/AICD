from flask import render_template
from flaskexample import app
from flask import request
import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from nltk.tokenize import RegexpTokenizer
import re
import string
import sklearn.preprocessing as pp
from scipy import sparse


def level2_title(c):
  if c['icd_level2'].startswith(
  ('0','10','11','12', '13')):
      return 'Infectious And Parasitic Diseases'
  elif c['icd_level2'].startswith(('1','20','21','22', '23')):
      return 'Neoplasms'
  elif c['icd_level2'].startswith(('24','25','26','27')):
      return 'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders'
  elif c['icd_level2'].startswith(('28')):
      return 'Diseases Of The Blood And Blood-Forming Organs'
  elif c['icd_level2'].startswith(('29','30','31')):
      return 'Mental Disorders'
  elif c['icd_level2'].startswith(('32','33', '34', '35', '36', '37', '38')):
      return 'Diseases Of The Nervous System And Sense Organs'
  elif c['icd_level2'].startswith(('39','40','41', '42', '43', '44', '45')):
      return 'Diseases Of The Circulatory System'
  elif c['icd_level2'].startswith(('46','47','48', '49', '40', '51')):
      return 'Diseases Of The Respiratory System'
  elif c['icd_level2'].startswith(('52','53','54', '55', '56', '57')):
      return 'Diseases Of The Digestive System'
  elif c['icd_level2'].startswith(('58','59', '60', '61', '62')):
      return 'Diseases Of The Genitourinary System'
  elif c['icd_level2'].startswith(('63','64','65', '66', '67')):
      return 'Complications Of Pregnancy, Childbirth, And The Puerperium'
  elif c['icd_level2'].startswith(('68','69','70')):
      return 'Diseases Of The Skin And Subcutaneous Tissue'
  elif c['icd_level2'].startswith(('71', '72', '73')):
      return 'Diseases Of The Musculoskeletal System And Connective Tissue'
  elif c['icd_level2'].startswith(('74', '75')):
      return 'Congenital Anomalies'
  elif c['icd_level2'].startswith(('76', '77')):
      return 'Certain Conditions Originating In The Perinatal Period'
  elif c['icd_level2'].startswith(('78', '79')):
      return 'Symptoms, Signs, And Ill-Defined Conditions'
  elif c['icd_level2'].startswith(('8', '9')):
      return 'Injury And Poisoning'
  elif c['icd_level2'].startswith(('V')):
      return 'Factors Influencing Health Status And Contact With Health Services'
  else:
    return 'External Causes Of Injury And Poisoning'

def preprocess_article_content(text_df):

    # tokenizer, stops, and stemmer
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words("english"))
    #add custom words
    stop_words.update(('and','I','A','And','So','arnt','This','When','It','many','Many','so','cant',
                       'Yes','yes','No','no','These','these', 'jpg', 'firstname', 'lastname', 'known'
                    'man', 'woman', 'old', 'year', 'name', 'note', 'nos', 'unspecifi', 'status', 'cmp','elsewher', 'type', 'without'))
    stemmer = SnowballStemmer('english')

    # process articles
    article_list = []
    for row, article in enumerate(text_df['text']):
        cleaned_tokens = []
        tokens = tokenizer.tokenize(article.decode('utf-8', 'ignore').lower())
        for token in tokens:
            if token not in stop_words:
                if len(token) > 1 and len(token) < 20: # removes non words
                    if not token[0].isdigit() and not token[-1].isdigit(): # removes numbers
                        stemmed_tokens = stemmer.stem(token)
                        cleaned_tokens.append(stemmed_tokens)
        # add process article
        article_list.append(' '.join(wd for wd in cleaned_tokens))

    return article_list

def bog_tf_idf (text_df):
    count = CountVectorizer(ngram_range=(2, 3), max_features=2000)
    #count = CountVectorizer(ngram_range=(2, 3))
    tfidf = TfidfTransformer()
    text_features = tfidf.fit_transform(count.fit_transform(np.array(text_df.textc))).toarray()
    vocab = count.get_feature_names()
    text_features = pd.DataFrame(data=text_features,
                                 columns=vocab)
    text_features = text_features.rename(columns = {'fit': 'fit_feature'}).reset_index(drop=True)
    return text_features

def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat



icd = pd.read_csv('./count_icd.csv', sep='\t', encoding='utf-8').replace(np.nan, '', regex=True)
icd.drop(icd.columns[[0]], axis=1, inplace=True)
df= pd.read_csv('./icd.csv', sep='\t', encoding='utf-8').replace(np.nan, '', regex=True)
df.drop(df.columns[[0]], axis=1, inplace=True)


@app.route('/')
@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/output')

def cesareans_output():
  #pull 'birth_month' from input field and store it
  df4 = df[['icd9_code','icd_level4','title']].groupby(["icd_level4"])["icd9_code", "title"].agg(lambda x: '; '.join(x.astype(str))).reset_index()
  t2=df4[['icd_level4','title']]
  t2.columns = ['icd_level4', 'text']
  n_icd = request.args.get('n_icd')
  n_icd = int(n_icd)
  n_codes = request.args.get('n_codes')
  n_codes = int(n_codes)
  inputtext = request.args.get('inputtext')
  ct = icd.iloc[0:n_icd,]
  ct['icd_level4']  = ct['icd9_code'].astype(str).str[0:4]
  t2 = pd.merge(ct, t2, on='icd_level4', how='left')[['icd_level4','text']].drop_duplicates()
  if inputtext:
      t2.loc[-1] = ['hadm_id', inputtext]# adding a row
      t2.index = t2.index + 1  # shifting index
      t2 = t2.sort_index()
      t2['textc']= preprocess_article_content(t2)
      textfeatures= bog_tf_idf(t2)
      newtext  = sparse.csr_matrix(np.asmatrix(textfeatures).T)
      sim =cosine_similarities(newtext).toarray()
      sim=sim[0,:]
      t2['sim'] = sim
      t2 =t2.sort_values('sim', ascending=False)
      siml = t2[t2.icd_level4 != "hadm_id"][['icd_level4','sim']].iloc[0:n_codes,:]
      siml_n= pd.merge(siml, df, on='icd_level4', how='left')[['icd_level4', 'long_title','sim', 'icd9_code']]
      siml_n['ICD_title'] = siml_n[['icd9_code', 'long_title']].apply(lambda x: ': '.join(x), axis=1)
      siml_n = siml_n[['icd_level4', 'sim', "ICD_title"]]
      s = 0.95/(siml_n.iloc[0]["sim"])
      siml_n['sim'] =s*siml_n['sim']
      siml_n.columns = ['ICD', 'Probability', 'Description']
      siml_n['idx'] = siml_n.groupby('ICD').cumcount() + 1
      siml_n['idx'] =siml_n['idx'].astype(str)
      siml_n = siml_n.round(4)
      Count_Row=siml_n.shape[0]
      return render_template("output.html", n_codes = n_codes, siml = siml_n, Count_Row = Count_Row )
  else: 
     
      siml_n = pd.DataFrame(columns=('ICD', 'Probability','Description'))
      for i in range (n_codes):
      	siml_n.loc[i] = 'No note input'
      
      return render_template("output.html", n_codes = n_codes, siml = siml_n, Count_Row = n_codes) 
      
      
      
