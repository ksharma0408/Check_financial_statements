from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import string
import nltk #natural Language Toolkit
nltk.download('stopwords') #contains is, am, are, he, she etc
nltk.download('wordnet') #contains the synonyms
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename,'rb'))
cv = pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    """
    
    df=pd.read_csv('C:\\Users\\shivam\\Downloads\\all-data.csv',names=['label','text'],encoding='latin-1')
    
    corpus=df['text'].values.tolist()
    
    nopunc=[]
    for i in range(len(corpus)):
      no  = [char for char in corpus[i] if char not in string.punctuation]
      nopunc.append(''.join(no))
    df['text_nopunc']=pd.Series(nopunc)
    
    stop=stopwords.words(fileids='english')
    text_nostop=[]
    for i in range(len(nopunc)):
        text_nostop.append(" ".join([word for word in str(nopunc[i]).split() if word not in stop]))
    df['text_nostop']=pd.Series(text_nostop)
    
    wordnet=WordNetLemmatizer()
    text_lemm=[]
    for i in range(len(text_nostop)):
        text_lemm.append(" ".join([wordnet.lemmatize(word) for word in text_nostop[i].split()]))
    df['text_lemm']=pd.Series(text_lemm)
    
    text_nonum=[]
    for i in range(len(text_lemm)):
        s=''
        for word in text_lemm[i].split():
          if word.isalpha():
            s+=word
          elif word.isnumeric():
            s+=num2words(int(word))
          s+=' ' 
        text_nonum.append(s)
    df['text_nonum']=pd.Series(text_nonum)
    
    df3=df[df['label']=='positive']
    
    df4=df[df['label']=='negative'] #class balancing
    df=pd.concat([df,df4,df3,df4,df4,df4])
    
    vect=CountVectorizer()
    X=vect.fit_transform(df.text_nonum)
    pickle.dump(vect,open('transform.pkl','wb'))
    
    vect = TfidfVectorizer()
    tfidf_matrix = vect.fit_transform(df['text_nonum'])
    df1= pd.DataFrame(tfidf_matrix.toarray(),columns=vect.get_feature_names())
    
    df['encoded_label'] = LabelEncoder().fit_transform(df['label'])
    
    x_train,x_test,y_train,y_test = train_test_split(df1,df.encoded_label,test_size = 0.25,random_state = 0)
    x_train.shape,x_test.shape,y_train.shape,y_test.shape
    
    model=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    filename = 'nlp_model.pkl'
    pickle.dump(model,open(filename,'wb'))
    """
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
