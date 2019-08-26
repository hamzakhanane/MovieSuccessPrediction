
##Authors: Hamza Khanane, Yonatan Cipriani

import csv
import matplotlib

import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from sklearn import tree
import pandas
from sklearn import preprocessing
from sklearn.externals.six import StringIO
from IPython.display import Image, display
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections
from sklearn.tree import export_graphviz
import graphviz

##opening the main data file and writing to a new with the extra column
with open('movie_metadata.csv','r') as csvinput:
    with open('output.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        ##adding an extra column to the output file
        for row in csv.reader(csvinput):
            if row[0] == 'color':
                writer.writerow(row+["Category"])
            else:
                ##intervals for flop,hit and blockbuster
                ##tried different experiments by changing our interval, this was the one which had the best accuracy rate
                float_item = float(row[28])
                if float_item < 5.5:
                    writer.writerow(row + ['Flop']) ##flop
                elif 5.5 < float_item < 7.5:
                    writer.writerow(row + ['Hit']) ##hit
                else:
                    writer.writerow(row + ['BlockBuster']) ##blockbuter
df = pd.read_csv('output.csv') ##making a data frame using the new output file with the new column

#the columns we cut out after feature selection
columns =  ["color", "genres","duration", "director_name", "language", "country", "content_rating", "actor_3_name",
            "actor_1_name",  "actor_2_name", "facenumber_in_poster","duration","num_user_for_reviews"
            ,"genres","movie_title","num_voted_users","plot_keywords"
            ,"movie_imdb_link","num_critic_for_reviews","title_year","imdb_score",
            "aspect_ratio","gross","actor_3_facebook_likes","imdb_score.1","movie_facebook_likes","budget"]

df = df.drop(columns,1) ##deleting all the other columns from our data frame
df = df.fillna(0) ##filling out the empyt values in our data frane with 0's

##preprocessing and cleaning the columns with the help of lable encoder. clearing all the noise and weird values which would hurt our decesion tree
string_columns = ["Category"]
le = preprocessing.LabelEncoder()
for col in string_columns:
	df[col] = le.fit_transform(df[col])
print(df)

##Our x includes all the other columns except for the category column
X = df.drop('Category', axis=1)
#y includes just category column, since this is the column we'll be writing our values to for testing
y = df['Category']


##code to create a graph which shows the correlations between columns


plt.matshow(df.corr())

col = ["director", "actor2","total-cast","actor1","Category"]
x_pos = np.arange(len(col))
plt.xticks(x_pos,col)
y_pos = np.arange(len(col))
plt.yticks(y_pos,col)

plt.show()


##splitting the data set into 80% training and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
##our clasifier with specific parameters
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=2, min_samples_leaf=6)
clf_gini.fit(X_train, y_train) ##fitting the test and the train sets
y_pred = clf_gini.predict(X_test)
y_pred

##printing the accuracy score
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)

##calcuating the precesion score
clf = SVC(random_state=42)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print ('Precision:', precision_score(y_test, y_pred, average='macro'))





##to make our visual representation of the decesion tree


data_feature_names = ["Category", "total cast likes", "director likes" , "actor1 likes"]

dot_data = tree.export_graphviz(clf_gini,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('dtree.png')

####################################################################################################################################################################################

#unused code which we wrote before when we were trying different algorithms also contains the kmeans clustering which we couldnt get to working unfortunately




# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)

# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn import preprocessing
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score
# import sklearn.cross_validation
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# df = pd.read_csv('movie_metadata.csv')
# columns = ['color','movie_imdb_link','aspect_ratio','language','duration','plot_keywords','country','content_rating','facenumber_in_poster','movie_title','title_year']
# #
#
# string_columns =  ["color", "genres","duration", "director_name", "language", "country", "content_rating", "actor_3_name", "actor_1_name",  "actor_2_name"]
# le = preprocessing.LabelEncoder()
# for col in string_columns:
# 	df[col] = le.fit_transform(df[col])
# df = df.drop(columns,1)  ##deleting the movie imdb link    ##deleting aspect_ratio cuz we have no idea what it is
# df = df.fillna(0)
# print(df)
# f1 = df['director_name'].values
# f2 = df['num_critic_for_reviews'].values
# f7 = df['genres'].values
# f8 = df['actor_1_name'].values
# f9 = df['num_voted_users'].values
# f11 = df['actor_3_name'].values
# f12 = df['num_user_for_reviews'].values
# f13 = df['budget'].values
# f16 = df['gross'].values
#
# X = np.array(list(zip(f1,f2,f7,f8,f9,f11,f12,f13,f16)))
#
# y = df['imdb_score'].values
# y.reshape(-1,1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# scalar = StandardScaler()
# scalar.fit(X_train)
# X_train = scalar.transform(X_train)
#
#
#
# k=6
# C_x = np.random.randint(0,np.max(X)-20,size=k)
# C_y = np.random.randint(0,np.max(X)-20,size=k)
# C = np.array(list(zip(C_x,C_y)),dtype=np.float32)
#
#
# kmeans = KMeans(n_clusters=k)
# kmeans = kmeans.fit(X_train,y_train)
#
# labels = kmeans.predict(X)
# centroids = kmeans.cluster_centers_
# print(kmeans.predict(y_test))
#
# # plt.scatter(f1,f2,c='#050505',s=20)
# # plt.scatter(C_x,C_y,marker='*',s=200,c='g')
# # plt.xlabel("Distance feature")
# # plt.ylabel("speeding feature")
# # plt.title('raw delviery fleet data')
# #
#
#
#
#
#
# # director_name
# # num_critic_for_reviews
# # director_facebook_likes
# # actor_3_facebook_likes
# # actor_2_name
# # actor_1_facebook_likes
# # gross
# # genres
# # actor_1_name
# # num_voted_users
# # cast_total_facebook_likes
# # actor_3_name
# # num_user_for_reviews
# # budget
# # actor_2_facebook_likes
# # movie_facebook_likes
# # imdb_score
#
#
#
#
#
#
#
#
# #
# # import pandas as pd
# # import numpy as np
# # from sklearn.cross_validation import train_test_split
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.metrics import accuracy_score
# # from sklearn import tree
# # import pandas
# # from sklearn import preprocessing
# #
# #
# #
# # df = pd.read_csv('movie_metadata.csv')
# #
# #
# #
# # ## deleting some of the features after making a data frame
# #
# # columns = ['color','movie_imdb_link','aspect_ratio','language','duration','plot_keywords','country','content_rating','facenumber_in_poster','movie_title','title_year']
# #
# # df = df.drop(columns,1)  ##deleting the movie imdb link    ##deleting aspect_ratio cuz we have no idea what it is
# #    ## deleting movie_imdb link cuz its useless
# #
# #
# # df = df.fillna(0)
# #
# # le = preprocessing.LabelEncoder()
# # le.fit(df)
# # print(df.head())
# #
# # X = df.drop('imdb_score', axis=1)
# # y = df['imdb_score']
# #
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# #
# # classifier = DecisionTreeClassifier()
# # classifier.fit(X_train, y_train)
# #
# #
# #
# #
# # # X = df.values[:,0:7]
# # # Y = df.values[:,8]
# # # ##print(Y)
# # # X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
# # # clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
# # #                                max_depth=3, min_samples_leaf=5)
# # # print(clf_gini.fit(X_train, y_train))
# #
