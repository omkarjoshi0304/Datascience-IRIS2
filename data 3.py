import pandas as pd
from  sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv("C:/Users/admin/Downloads/IRIS.csv")

rf=RandomForestClassifier(random_state=1)

x= df.drop("species",axis=1)
y= df["species"]

bestfeatures=SelectKBest(score_func=chi2,k="all")
fit=bestfeatures.fit(x,y)
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)
featuresScores = pd.concat([dfcolumns,dfscores],axis=1)
featuresScores.columns=['Specs','score']

print(featuresScores)

model= ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)

feat_importance=pd.Series(model.feature_importances_, index=x.columns)
feat_importance.nlargest(4).plot(kind='barh')
plt.show()

