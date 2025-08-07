import json
import pandas as pd
import re
import joblib


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()
data = []
with open(r'D:\Personal Project\text-topic-classifier\data\News_Category_Dataset_v3.json','r') as f:
    for line in f:
        data.append(json.loads(line))

data = pd.DataFrame(data)
category_mapping = {
    'ARTS & CULTURE': 'ARTS',
    'CULTURE & ARTS': 'ARTS',

    'PARENTS': 'PARENTING',

    'WORLDPOST': 'WORLD',
    'THE WORLDPOST': 'WORLD',
    'WORLD NEWS': 'WORLD',

    'STYLE & BEAUTY': 'STYLE',

    'HOME & LIVING': 'LIFESTYLE',
    'WELLNESS': 'LIFESTYLE',
    'HEALTHY LIVING': 'LIFESTYLE',

    'FOOD & DRINK': 'FOOD',
    'TASTE': 'FOOD',

    'GREEN': 'ENVIRONMENT',

    'COLLEGE': 'EDUCATION',

    'MONEY': 'BUSINESS',

    'LATINO VOICES': 'IDENTITY',
    'QUEER VOICES': 'IDENTITY',
    'BLACK VOICES': 'IDENTITY',
    'WOMEN': 'IDENTITY',

    'U.S. NEWS': 'POLITICS',
}

data['category'] = data['category'].replace(category_mapping)
data['text'] = data['headline'] + '. ' + data['short_description']
data = data[['text','category']]
data['text'] = data['text'].apply(clean_text)

le = LabelEncoder()
x = data['text']
y = le.fit_transform(data['category'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)


cls = ImbPipeline(steps=[
    ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.9)),
    ('oversample', RandomOverSampler(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=-1)),

])
cls.fit(x_train, y_train)

y_pred = cls.predict(x_test)
print(classification_report(
    le.inverse_transform(y_test),
    le.inverse_transform(y_pred),
    zero_division=0
))

joblib.dump(cls, r"D:\Personal Project\text-topic-classifier\outputs\text_classifier.pkl")
joblib.dump(le, r"D:\Personal Project\text-topic-classifier\outputs\label_encoder.pkl")
print("✅ Đã lưu mô hình và label encoder.")