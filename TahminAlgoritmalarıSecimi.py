import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Veri Yükleme ve Ön İşleme
orginalVeriler = pd.read_csv('studentMat.csv')
copyVeriler = orginalVeriler.copy()

le = preprocessing.LabelEncoder()

# Label encoder uygulaması
def label_encoder(a, b):
    while a < b:
        copyVeriler.iloc[:, a] = le.fit_transform(copyVeriler.iloc[:, a])
        a += 1

label_encoder(0, 2)
label_encoder(3, 6)
label_encoder(8, 9)
label_encoder(12, 20)

# Bağımlı ve bağımsız değişkenlerin oluşturulması
sinavSonuc = copyVeriler.iloc[:, -1]
sinavSonucDf = pd.DataFrame(data=sinavSonuc, index=range(395), columns=['exam_avg'])

tumParametreler = copyVeriler.iloc[:, 0:27]
tumParametrelerDf = pd.DataFrame(data=tumParametreler, index=range(395), columns=[
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
    'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 
    'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
])

# Eğitim ve test verisi olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(tumParametrelerDf, sinavSonucDf, train_size=0.8, random_state=0)

# Modellerin oluşturulması ve değerlendirilmesi
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (Degree 2)": PolynomialFeatures(degree=2),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100, random_state=0),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=0)
}

r2_scores = {}

# Lineer Regresyon
linear_model = models["Linear Regression"]
linear_model.fit(x_train, y_train)
y_pred = linear_model.predict(x_test)
r2_scores["Linear Regression"] = r2_score(y_test, y_pred)

# Polinom Regresyon
poly = models["Polynomial Regression (Degree 2)"]
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)
poly_model = LinearRegression()
poly_model.fit(x_poly_train, y_train)
y_pred = poly_model.predict(x_poly_test)
r2_scores["Polynomial Regression (Degree 2)"] = r2_score(y_test, y_pred)

# KNN Regresyon
knn_model = models["KNN Regression"]
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)
r2_scores["KNN Regression"] = r2_score(y_test, y_pred)

# Random Forest Regresyon
rf_model = models["Random Forest Regression"]
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
r2_scores["Random Forest Regression"] = r2_score(y_test, y_pred)

# Karar Ağacı Regresyon
dt_model = models["Decision Tree Regression"]
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)
r2_scores["Decision Tree Regression"] = r2_score(y_test, y_pred)

print(r2_scores)



