import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
speaker_gender = pd.read_csv('ted_speaker_gender.csv')

speaker_gender['he_she_count'] = speaker_gender['he_count'] > speaker_gender['she_count']
speaker_gender['he_she'] = speaker_gender['he_she_count'].map(lambda x: int(x))

# Splitting the dataset between training and testing set
msk = np.random.rand(len(speaker_gender)) < 0.8
train_df = speaker_gender.loc[msk]
test_df = speaker_gender.loc[~msk]

X_train = train_df['name_in_profile']
Y_train = train_df['he_she']
X_test = test_df['name_in_profile']
Y_test = test_df['he_she']


from sklearn.base import TransformerMixin


class ExtractNames(TransformerMixin):
    def transform(self, X, *args):
        lst = []
        for name in X:
            name = name.lower()
            first = name.split()[0]
            last = name.split()[-1]
            lst.append({
                'last_letter': first[-1],
                'last_two': first[-2:],
                'last_three': first[-3:],
                'last_is_vowel': (first[-1] in 'AEIOUYaeiouy')
            })
        return lst

    def fit(self, *args):
        return self


trans = ExtractNames()
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline


t_sne = make_pipeline(ExtractNames(), DictVectorizer())
clf0 = t_sne.fit_transform(speaker_gender["name_in_profile"])

from sklearn.manifold import TSNE
tsne_model = TSNE(perplexity=40, n_components=2,
                  init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(clf0.todense())


import seaborn as sns
colors = sns.mpl_palette("Dark2", 7)
points = np.array(new_values)
import matplotlib.pyplot as plt

colors_p = speaker_gender['he_she'].map(lambda x: colors[x]).tolist()
for i in range(len(points)):
    plt.scatter(points[i, 0], points[i, 1], color=colors_p[i])

# plt.colorbar()
plt.savefig('t_sne_2d.png', dpi=200)
plt.clf()
plt.cla()
plt.close()


tsne_model_3d = TSNE(perplexity=40, n_components=3,
                     init='pca', n_iter=2500, random_state=23)
new_values_3d = tsne_model_3d.fit_transform(clf0.todense())
points_3d = np.array(new_values_3d)
from mpl_toolkits.mplot3d import axes3d, Axes3D
fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection = '3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
           c=speaker_gender['he_she'].tolist(), cmap='Dark2')
plt.savefig('t_sne_3d.png', dpi=200)
plt.clf()
plt.cla()
plt.close()


from sklearn.linear_model import LogisticRegressionCV
clf = make_pipeline(ExtractNames(), DictVectorizer(), LogisticRegressionCV())
clf.fit(X_train, Y_train)
test_df['predicted'] = clf.predict(test_df['name_in_profile'])
print("LogisticRegressionCV score : {}".format(clf.score(X_test, Y_test)))
# LogisticRegressionCV score : 0.7746478873239436

clf1 = make_pipeline(ExtractNames(), DictVectorizer(),
                     RandomForestClassifier(n_estimators=100))
clf1.fit(X_train, Y_train)
test_df['predicted'] = clf1.predict(test_df['name_in_profile'])
print("RandomForestClassifier score : {}".format(clf1.score(X_test, Y_test)))
# RandomForestClassifier score : 0.7535211267605634
# https://github.com/sholiday/genderPredictor/blob/master/genderPredictor.py
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import MultinomialNB
clf2 = make_pipeline(ExtractNames(), DictVectorizer(), SelectKBest(k=5, score_func=f_classif),
                     MultinomialNB())
clf2.fit(X_train, Y_train)
test_df['predicted'] = clf2.predict(test_df['name_in_profile'])
print("MultinomialNB score : {}".format(clf2.score(X_test, Y_test)))
# MultinomialNB score : 0.7464788732394366
