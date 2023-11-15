import pandas as pd
import re
import nltk
import pickle
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
ps=WordNetLemmatizer()
#ph=PorterStemmer()
#print(stopwords.words('english'))
data=pd.read_csv('C:/Users/Welcome/Desktop/spam.csv',encoding="latin-1")
data.dropna(how="any", inplace=True, axis=1)
data.columns = ['label', 'message']
#data.head(20)
c=[]
for i in range(len(data)):
    review=re.sub("[^a-zA-Z]"," ",data['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.lemmatize(j) for j in review if j not in stopwords.words('english')]
    review=" ".join(review)
    c.append(review)
from sklearn.feature_extraction.text import CountVectorizer
tfidf=CountVectorizer()
X=tfidf.fit(c)
X=tfidf.fit_transform(c)

pickle.dump(tfidf,open('transform.pkl','wb'))
df=pd.DataFrame(X.toarray(),columns=tfidf.get_feature_names())
print(df)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X.toarray(),data['label'],test_size=0.3,random_state=0)

from sklearn.naive_bayes import MultinomialNB
mul=MultinomialNB().fit(X_train, y_train)
y_pred=mul.predict(X_test)
filename='spam_model.pkl'
pickle.dump(mul,open(filename,'wb'))
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)