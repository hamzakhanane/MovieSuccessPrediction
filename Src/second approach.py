import csv
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas
from sklearn import preprocessing
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv('output.csv')

# columns =  ["color", "genres","duration", "director_name", "language", "country", "content_rating", "actor_3_name",
            # "actor_1_name",  "actor_2_name", 'facenumber_in_poster',"duration","director_facebook_likes",
            # "actor_1_facebook_likes","genres","movie_title","cast_total_facebook_likes","plot_keywords"
            # ,"movie_imdb_link","budget","title_year","actor_2_facebook_likes","imdb_score","aspect_ratio","gross","actor_3_facebook_likes","imdb_score.1"]
##list of columns which we removed from the data set
col2=  ["color","duration", "director_name", "language", "country", "content_rating", "actor_3_name",
            "actor_1_name",  "actor_2_name", 'facenumber_in_poster',"duration","movie_title","plot_keywords"
            ,"movie_imdb_link","title_year","imdb_score","aspect_ratio","gross","actor_3_facebook_likes","imdb_score.1", "num_user_for_reviews", "num_critic_for_reviews", "movie_facebook_likes", "num_voted_users", "genres", "budget"]
			
df = df.drop(col2,1) ##droping the columns
df = df.fillna(0) ##filling the empty values with 0's


string_columns = ["Category"] ##preprocess the category column
le = preprocessing.LabelEncoder()
for col in string_columns:
	df[col] = le.fit_transform(df[col])
print(df)

##taking all the columns in the data frame except for the category
X = df.drop('Category', axis=1)
y = df['Category']
##splitting the dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=2, min_samples_leaf=5)

clf = SVC(random_state=42) ##support vectorm machine score
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print ('Precision:', precision_score(y_test, y_pred, average='macro'))
print(list(df))
