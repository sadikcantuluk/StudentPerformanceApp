

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:06:31 2024

@author: sadik
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFormLayout, QMessageBox
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

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

# Sınıf etiketlerini oluşturma
# Burada 10 eşiğini belirledik, istediğiniz değere değiştirebilirsiniz
copyVeriler['success'] = copyVeriler['exam_avg'].apply(lambda x: 1 if x >= 10 else 0)

# Bağımlı ve bağımsız değişkenlerin oluşturulması
sinavSonuc = copyVeriler.iloc[:, -2]
success = copyVeriler['success']
tumParametreler = copyVeriler.iloc[:, 0:27]

# Eğitim ve test verisi olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(tumParametreler, sinavSonuc, train_size=0.8, random_state=0)
x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(tumParametreler, success, train_size=0.8, random_state=0)

# Random Forest modelinin eğitilmesi (Regresyon)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
rf_regressor.fit(x_train, y_train)

# Random Forest modelinin eğitilmesi (Sınıflandırma)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(x_train_clf, y_train_clf)

# Ortalama değerleri hesaplama
mean_values = tumParametreler.mean()

class StudentPerformancePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Öğrenci Başarısını Tahmin Et')
        
        # Form düzeni
        form_layout = QFormLayout()
        
        # Kullanıcıdan alınacak parametreler
        self.inputs = {}
        params = {
            'Cinsiyet (K=1, E=0)': 'sex',
            'Yaş': 'age',
            'Adres (Kentsel=1, Kırsal=0)': 'address',
            'Aile Büyüklüğü (GT3=1, LE3=0)': 'famsize',
            'Ebeveyn Durumu (Birlikte=1, Ayrı=0)': 'Pstatus',
            'Anne Eğitimi (0-4)': 'Medu',
            'Baba Eğitimi (0-4)': 'Fedu',
            'Seyahat Süresi (1-4)': 'traveltime',
            'Çalışma Süresi (1-4)': 'studytime',
            'Ek Eğitim Desteği (Evet=1, Hayır=0)': 'schoolsup',
            'Aktivitelere Katılım (Evet=1, Hayır=0)': 'activities',
            'Yüksek Eğitim İsteği (Evet=1, Hayır=0)': 'higher',
            'İnternet Erişimi (Evet=1, Hayır=0)': 'internet',
            'Romantik İlişki (Evet=1, Hayır=0)': 'romantic',
            'Aile İlişkileri (1-5)': 'famrel',
            'Hafta İçi Alkol Tüketimi (1-5)': 'Dalc',
            'Hafta Sonu Alkol Tüketimi (1-5)': 'Walc',
            'Sağlık Durumu (1-5)': 'health'
        }
        
        for label, key in params.items():
            self.inputs[key] = QLineEdit()
            form_layout.addRow(QLabel(label), self.inputs[key])
        
        # Tahmin butonu
        self.predict_btn = QPushButton('Tahmin Et')
        self.predict_btn.clicked.connect(self.predict)
        
        # Düzenlerin ayarlanması
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.predict_btn)
        
        self.setLayout(main_layout)
    
    def predict(self):
        # Kullanıcıdan alınan değerler
        input_data = []
        params_order = [
            'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'guardian', 'traveltime',
            'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
            'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences'
        ]
        
        for param in params_order:
            if param in self.inputs:
                value = self.inputs[param].text()
                if value == '':
                    value = mean_values[param]
                input_data.append(float(value))
            else:
                input_data.append(mean_values[param])
        
        input_data = np.array(input_data).reshape(1, -1)
        
        # Not tahmini yapma
        exam_avg_prediction = rf_regressor.predict(input_data)[0]
        
        # Başarı tahmini yapma
        success_prediction = rf_classifier.predict(input_data)[0]
        success_status = 'Başarılı' if success_prediction == 1 else 'Başarısız'
        
        # Sonucu gösterme
        QMessageBox.information(self, 'Tahmin Sonucu', f'Tahmin Edilen Not: {exam_avg_prediction:.2f}\nBaşarı Durumu: {success_status}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = StudentPerformancePredictor()
    ex.show()
    sys.exit(app.exec_())
