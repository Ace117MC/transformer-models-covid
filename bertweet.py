# Install Required Libraries if required
!pip install simpletransformers
!pip install tensorboardX

# Importing Libraries

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import logging
import pandas as pd
import re
import sklearn



# Helper Functions

def clean_text(text):
    text = str(text)
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"#[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub(r"[0-9]", '', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text.lower()

# Read CSV File
tweets = pd.read_csv('covid-19_vaccine_tweets_with_sentiment.csv', encoding='unicode_escape')
print("Data Points:", len(tweets))
print(tweets.head())
df = tweets[['tweet_text','label']]

# Preprocessing Data
text = df.tweet_text.apply(clean_text)
df['tweet_text'] = text

# Train Test Split
df, df_test = train_test_split(df, test_size=0.2, random_state=42)
print("Training Samples:", len(df), "\nTesting Samples:", len(df_test))

# Model Definition
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Config
model_args = ClassificationArgs(num_train_epochs=5)
model_args.overwrite_output_dir = True
model_args.manual_seed = 4
model_args.use_multiprocessing = True
model_args.train_batch_size = 16
model_args.eval_batch_size = 8
model_args.labels_list = [1, 2, 3]
model_args.fp16 = False
model_args.gradient_accumulation_steps = 2

model = ClassificationModel(
    'bertweet',
    'vinai/bertweet-base',
    num_labels=3,
    args=model_args,
    use_cuda=True,
) 

# Model Training
model.train_model(df)

# Evaluate Model
result, model_outputs, wrong_predictions = model.eval_model(df_test, acc=sklearn.metrics.accuracy_score, cls_rpt = sklearn.metrics.classification_report)

print("Accuracy:", result['acc'])
print("Phi Coefficient:", result['mcc'])
print(result['cls_rpt'])

