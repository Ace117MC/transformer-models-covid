# Install Libraries if required
!pip install sentencepiece
!pip install transformers

# Importing Libraries

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, TFBertForSequenceClassification

import nltk
import numpy as np
import pandas as pd
import re
import tensorflow as tf

nltk.download('stopwords')
nltk.download('punkt')

# Helper Functions

def text_preprocessing(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub('[/(){}\[\]\|@,.;_]', ' ', text)
    return text.lower()


# Read CSV File
data = pd.read_csv('covid-19_vaccine_tweets_with_sentiment.csv', encoding='unicode_escape')
print("Data Points:", len(data))
data.columns = ['id', 'label', 'tweet']
print(data.head())

# Preprocess Data
data['tweet'] = data['tweet'].apply(text_preprocessing)
data['label'] = data['label']-1

# Split Train and Test Data
data, test_data = train_test_split(data, test_size=0.2, random_state=518)
print("Training Samples:", len(data), "\nTesting Samples:", len(test_data))

# Model Definition
base_covid_bert_model = 'digitalepidemiologylab/covid-twitter-bert-v2'
bert_tokenizer = BertTokenizer.from_pretrained(base_covid_bert_model)
bert_model = TFBertForSequenceClassification.from_pretrained(base_covid_bert_model,num_labels=3)

# Converting input text in the tokens form
print("Converting Input Text to Tokens")
# Train Data
sentences = data['tweet']
labels = data['label']
input_ids=[]
attention_masks=[]

for sent in sentences:
    bert_inp=bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =128,pad_to_max_length = True,return_attention_mask = True)
    input_ids.append(bert_inp['input_ids'])
    attention_masks.append(bert_inp['attention_mask'])

input_ids=np.asarray(input_ids)
attention_masks=np.array(attention_masks)
labels=np.array(labels)

# Test Data
sentences_test = test_data['tweet']
y_test = test_data['label']
X_test=[]
mask_test=[]

for sent in sentences_test:
    bert_inp=bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =128,pad_to_max_length = True,return_attention_mask = True)
    X_test.append(bert_inp['input_ids'])
    mask_test.append(bert_inp['attention_mask'])

X_test=np.asarray(X_test)
mask_test=np.array(mask_test)
y_test=np.array(y_test)

# Splitting Train and Validation Data
X_train,X_val,y_train,y_val,mask_train,mask_val=train_test_split(input_ids,labels,attention_masks,test_size=0.15)

# Compiling CT-Bert Model
print('\nBert Model',bert_model.summary())
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)
bert_model.compile(loss=loss,optimizer=optimizer,metrics=[metric])

# Training Model
history=bert_model.fit([X_train,mask_train],y_train,batch_size=8,epochs=4,validation_data=([X_val,mask_val],y_val))

#Evaluating on test data
y=bert_model.predict([X_test,mask_test])
y_pred = np.argmax(y['logits'],axis=1)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, y_pred, average='macro')
print('Precision: %f' % precision)
recall = recall_score(y_test, y_pred,average='macro')
print('Recall: %f' % recall)
f1 = f1_score(y_test, y_pred,average='macro')
print('F1 score: %f' % f1)
print(classification_report(y_test, y_pred))