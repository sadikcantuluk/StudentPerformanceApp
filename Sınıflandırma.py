# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:35:12 2024

@author: sadik
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Veri Yükleme ve Ön İşleme
orginalVeriler = pd.read_csv('studentMat.csv')
copyVeriler = orginalVeriler.copy()

le = LabelEncoder()

# Label encoder uygulaması
def label_encoder(a, b):
    while a < b:
        copyVeriler.iloc[:, a] = le.fit_transform(copyVeriler.iloc[:, a])
        a += 1

label_encoder(0, 2)
label_encoder(3, 6)
label_encoder(8, 9)
label_encoder(12, 20)

# Sınıf etiketlerini oluşturma
copyVeriler['success'] = copyVeriler['exam_avg'].apply(lambda x: 1 if x >= 10 else 0)

# Bağımlı ve bağımsız değişkenlerin oluşturulması
X = copyVeriler.iloc[:, 0:27]
y = copyVeriler['success']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Modellerin oluşturulması
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='linear')
}

# Modellerin eğitilmesi ve değerlendirilmesi
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    results[model_name] = {
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }

# En iyi performans gösteren modelin seçilmesi
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]

print(f"En iyi model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.2f}")
print("Confusion Matrix:")
print(results[best_model_name]['confusion_matrix'])
print("Classification Report:")
print(results[best_model_name]['classification_report'])



































