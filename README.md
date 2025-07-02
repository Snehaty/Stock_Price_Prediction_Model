# 📈 LSTM Stock Price Prediction & Forecasting Dashboard

This project leverages deep learning (LSTM networks) to forecast stock prices, featuring an interactive dashboard for effective visualization and analysis. It demonstrates the application of machine learning in financial time series forecasting and showcases end-to-end engineering, from data preprocessing to deployment.

---

## 🚀 Overview

- **Objective:** Predict future stock prices using historical data with an LSTM neural network.
- **Features:**
  - Feature engineering (SMA, EMA, momentum, etc.)
  - LSTM model training and evaluation
  - Interactive Streamlit dashboard for real-time prediction and visualization
  - Clear comparison of actual vs. predicted prices, rolling forecasts, and residual analysis

---



## 🛠️ Tech Stack

- **Programming:** Python 3.x
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras, Streamlit
- **Tools:** Git, GitHub, Jupyter Notebook, ngrok (for deployment)

---

## 📊 Key Features

- **End-to-End Pipeline:**  
  Data loading → Cleaning → Feature engineering → Train/test split → Model training → Evaluation → Deployment.
- **Deep Learning Model:**  
  LSTM layers capture temporal dependencies in financial data for robust predictions.
- **Dashboard:**  
  Built with Streamlit for user-friendly interaction. Plots actual vs. predicted prices, forecasts, and error metrics.
- **Customizable:**  
  Users can select the stock, adjust look-back windows, and tweak forecast parameters.

---



- **Performance Metrics:**  
  - RMSE, MAE, and R² displayed on dashboard
  - Visual inspection of forecast accuracy via plots

---

## 📂 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Snehaty/lstm-stock-dashboard.git
   cd lstm-stock-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit dashboard**
   ```bash
   streamlit run app.py
   ```

4. **(Optional) Expose the dashboard online**
   ```bash
   ngrok http 8501
   ```
   Use the provided ngrok URL to access remotely.

---

## 🧠 What You'll Learn

- How to preprocess time series financial data for ML
- Building and tuning LSTM models for sequence prediction
- Deploying ML models as web apps with Streamlit
- Visualizing forecasts and understanding limitations of time series prediction

---

## 🏆 Why This Project?

Financial forecasting is a cornerstone of data-driven engineering in industry. This project demonstrates:
- **Technical depth:** Advanced ML and deep learning for real-world data
- **Software engineering:** Modular, reproducible code, version control, and deployment
- **Communication:** Clear results, visualizations, and documentation

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome!  
If you use this project, please ⭐️ the repo and share your feedback.

---

## 📬 Contact

- [GitHub](https://github.com/Snehaty)
- [LinkedIn](https://linkedin.com/in/sneha-tyagi-482692289)
- [Email](mailto:snehatyagi4002@gmail.com)

---

> _“Turning data into action: forecasting tomorrow’s markets, today.”_
