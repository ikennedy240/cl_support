import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/seattle_sample_kwd.csv")

n_train = int(df.shape[0]*0.75)
n_test = df.shape[0]-n_train
df_train = df.ix[0:(n_train-1)]
df_test = df.ix[n_train:(n_train+n_test-1)]

cv = CountVectorizer(stop_words="english")
counts_train = cv.fit_transform(df_train.ix[:]["listingText"])
tf_xform = TfidfTransformer(use_idf=False).fit(counts_train)
tfidf_train = tf_xform.fit_transform(counts_train)
cfNB = MultinomialNB().fit(tfidf_train, df_train.ix[:]["unit_type"])

counts_test = cv.transform(df_test.ix[:]["listingText"])
tfidf_test = tf_xform.transform(counts_test)
pred = cfNB.predict(tfidf_test)

print(cv.vocabulary_)
print(pred)
