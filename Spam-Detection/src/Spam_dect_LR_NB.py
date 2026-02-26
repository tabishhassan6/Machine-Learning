import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import  MultinomialNB
df = pd.read_csv(r"Spam\spam.csv",encoding="latin1")
print("First 10 rows of dataframe:\n",df.head(10))
print("Info of dataframe:\n",df.info())
print("Shape of dataframe:\n",df.shape)
print("Columns:\n",df.columns)
print("Describe content:\n",df.describe())

df = df[["v1","v2"]]
print(df.head(10))

df = df.rename(columns={"v1":"Label","v2":"Message"})
print(df.head(10))

df["Label"]= df["Label"].map({"ham":0,"spam":1})
print(df.head(10))
X =df["Message"]
y=df["Label"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

cv = CountVectorizer()
X_train = cv.fit_transform(X_train) #Learn + Transform
X_test = cv.transform(X_test)#Transform bss

model = LogisticRegression()
model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print("Accuracy using Logistic Regression:",accuracy)

y_pred = model.predict(X_test)

cm =confusion_matrix(y_test,y_pred)
print(cm)

print(classification_report(y_test,y_pred))
# SECOND MODEL NAIVE BAYES

model = MultinomialNB()
model.fit(X_train,y_train)

accuracy = model.score(X_test,y_test)
print("Accuracy using Naive Bayes:",accuracy)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

