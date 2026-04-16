# 💤 Sleep Efficiency Analysis and Insomnia Prediction

## 📌 Project Overview
This project predicts **sleep quality and possible insomnia** using Machine Learning based on lifestyle habits.

It analyzes user inputs such as sleep duration, caffeine intake, exercise, and more to classify:
- ✅ Good Sleep Quality  
- ⚠️ Poor Sleep (Possible Insomnia)

---

## 🎯 Objective
To build a machine learning model that can help identify sleep disorders early and provide useful insights for better sleep health.

---

## 📊 Dataset
- Source: Kaggle Sleep Efficiency Dataset  
- Features include:
  - Age  
  - Gender  
  - Sleep Duration  
  - Awakenings  
  - Caffeine Consumption  
  - Alcohol Consumption  
  - Exercise Frequency  
  - Smoking Status  
  - Sleep Efficiency  

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  

---

## 🤖 Machine Learning Models
- Decision Tree Classifier  
- Random Forest Classifier (Final Model)

---

## 🧠 Model Workflow
1. Data preprocessing (handling missing values, encoding)
2. Feature selection
3. Creating target variable (Sleep Quality)
4. Train-test split
5. Model training
6. Model evaluation
7. Prediction on new data

---

## 📈 Model Performance
- Decision Tree Accuracy: ~89%  
- Random Forest Accuracy: ~93% ✅  

---

## 🌐 Web Application
Built using **Streamlit** for interactive prediction.

Users can:
- Enter their details
- Get instant prediction
- View sleep quality results

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
