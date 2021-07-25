import numpy as np
from flask import Flask, request, render_template
import joblib
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model , load_model
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import one_hot
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)                   #creating the Flask class object 
model = load_model('Prediction.h5')    
#tfidf = joblib.load(open('tfidf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

voc_size = 50000
def preprocess(tweet):
    
    ps=PorterStemmer()
    
    L=[]

    review = re.sub('[^a-zA-Z]','',tweet[0])
    review=review.lower()
    review=review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english') ]
    review=''.join(review)
    L.append(review)
    
    oe=[one_hot(words,voc_size) for words in L]
    sent_length=250
    doc = pad_sequences(oe,padding='pre',maxlen = sent_length)
    
    return doc
    
dic={0:'INFP',1:'INFJ',2:'INTP',3:'INTJ ',4:'ENTP ',5:'ENFP ',6:'ISTP',7:'ISFP ',8:'ENTJ ',9:'ISTJ ',10:'ENFJ',11:'ISFJ ',12:'ESTP ',13:'ESFP',14:'ESFJ ',15:'ESTJ'}  

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   


    if request.method == 'POST':
    
        text = request.form['Review']
        data = [text]
        vectorizer = preprocess(data)
        prediction = model.predict(vectorizer)
        prediction=np.argmax(prediction)
        pr=dic[prediction]
        result =str(pr)         
        
        return render_template('index.html', prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
