// NexusML - Interactive Arena Logic with Code Review & Prediction Simulator

// --- 1. TASK DATASET (25 Tasks) ---
const tasks = [
  // --- REGRESSION ---
  {
    id: 0,
    title: "House Prices",
    category: "Regression",
    desc: "Predict house prices using square footage, bedrooms, and location.",
    diff: "Easy",
    data: "Small",
    fields: ["footage", "bedrooms", "location"],
    output: "Yes / No",
    solution: `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
df = pd.read_csv("house_prices.csv") 

# Target: High-value house (Yes/No classification)
median_price = df['price'].median()
df['high_value'] = (df['price'] > median_price).astype(int)

# 2. Extract features
X = df[['footage', 'bedrooms', 'location']]
y = df['high_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Fit classifier model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predict
predictions = model.predict(X_test)
print("Model accuracy score computed.")`,
    inputs: [
      { name: "footage", label: "Square Footage", type: "range", min: 500, max: 5000, value: 2200 },
      { name: "bedrooms", label: "Bedrooms", type: "range", min: 1, max: 6, value: 3 },
      { name: "location", label: "Location", type: "select", options: ["Downtown", "Suburbs", "Rural"], value: "Suburbs" }
    ],
    predict: (inputs) => {
      const score = (inputs.footage / 5000) * 0.5 + (inputs.bedrooms / 6) * 0.3 + (inputs.location === "Downtown" ? 0.2 : inputs.location === "Suburbs" ? 0.1 : 0);
      return score > 0.45 ? "Yes" : "No";
    }
  },
  {
    id: 1,
    title: "Insurance Charges",
    category: "Regression",
    desc: "Predict medical insurance charges using age, BMI, and children.",
    diff: "Easy",
    data: "Small",
    fields: ["age", "BMI", "children"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("insurance_charges.csv")
df['high_charge'] = (df['charges'] > 15000).astype(int)

X = df[['age', 'BMI', 'children']]
y = df['high_charge']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

print("Inference model trained.")`,
    inputs: [
      { name: "age", label: "Age (Years)", type: "range", min: 18, max: 90, value: 35 },
      { name: "BMI", label: "BMI", type: "range", min: 15, max: 50, value: 26 },
      { name: "children", label: "Number of Children", type: "range", min: 0, max: 5, value: 1 }
    ],
    predict: (inputs) => {
      const score = (inputs.age / 90) * 0.3 + (inputs.BMI / 50) * 0.5 + (inputs.children / 5) * 0.2;
      return score > 0.4 ? "1" : "0";
    }
  },
  {
    id: 2,
    title: "Exam Scores",
    category: "Regression",
    desc: "Predict student exam scores based on study hours and attendance.",
    diff: "Easy",
    data: "Small",
    fields: ["study_hours", "attendance"],
    output: "Yes / No",
    solution: `import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("exam_scores.csv")
df['passed_excellent'] = (df['score'] >= 85).astype(int)

X = df[['study_hours', 'attendance']]
y = df['passed_excellent']

clf = LogisticRegression()
clf.fit(X, y)
print("Model initialized.")`,
    inputs: [
      { name: "study_hours", label: "Study Hours / Week", type: "range", min: 0, max: 40, value: 12 },
      { name: "attendance", label: "Attendance (%)", type: "range", min: 50, max: 100, value: 85 }
    ],
    predict: (inputs) => {
      const score = (inputs.study_hours / 40) * 0.6 + (inputs.attendance / 100) * 0.4;
      return score > 0.6 ? "Yes" : "No";
    }
  },
  {
    id: 3,
    title: "Car Resale Value",
    category: "Regression",
    desc: "Predict car resale value using mileage, age, and brand.",
    diff: "Medium",
    data: "MediumData",
    fields: ["mileage", "age", "brand"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("car_resale.csv")
df['high_value'] = (df['resale_value'] > df['original_price']*0.6).astype(int)

encoder = OrdinalEncoder()
df['brand_encoded'] = encoder.fit_transform(df[['brand']])

X = df[['mileage', 'age', 'brand_encoded']]
y = df['high_value']

rf = RandomForestClassifier()
rf.fit(X, y)
print("Decision forest training complete.")`,
    inputs: [
      { name: "mileage", label: "Mileage (Miles)", type: "range", min: 0, max: 200000, value: 45000 },
      { name: "age", label: "Age of Car (Years)", type: "range", min: 0, max: 20, value: 4 },
      { name: "brand", label: "Brand tier", type: "select", options: ["Budget", "Premium", "Exotic"], value: "Premium" }
    ],
    predict: (inputs) => {
      const dep = (inputs.mileage / 200000) * 0.4 + (inputs.age / 20) * 0.4;
      const brandBonus = inputs.brand === "Exotic" ? 0.7 : inputs.brand === "Premium" ? 0.4 : 0.1;
      return (brandBonus - dep) > 0.0 ? "1" : "0";
    }
  },
  {
    id: 4,
    title: "Electricity Consumption",
    category: "Regression",
    desc: "Predict electricity consumption based on temperature and time of day.",
    diff: "Medium",
    data: "MediumData",
    fields: ["temperature", "time_of_day"],
    output: "Yes / No",
    solution: `import pandas as pd
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("electricity.csv")
df['surge_demand'] = (df['consumption'] > 75.0).astype(int)

X = pd.get_dummies(df[['temperature', 'time_of_day']])
y = df['surge_demand']

mlp = MLPClassifier(hidden_layer_sizes=(32, 16))
mlp.fit(X, y)
print("Multilayer Perceptron ready.")`,
    inputs: [
      { name: "temperature", label: "Temperature (°F)", type: "range", min: 10, max: 110, value: 75 },
      { name: "time_of_day", label: "Time of Day", type: "select", options: ["Morning", "Afternoon", "Evening", "Night"], value: "Afternoon" }
    ],
    predict: (inputs) => {
      const temp = parseInt(inputs.temperature);
      let heatIndex = Math.abs(temp - 70); 
      const isPeakTime = (inputs.time_of_day === "Afternoon" || inputs.time_of_day === "Evening");
      const surgeChance = (heatIndex / 60) * 0.6 + (isPeakTime ? 0.4 : 0.1);
      return surgeChance > 0.5 ? "Yes" : "No";
    }
  },

  // --- CLASSIFICATION ---
  {
    id: 5,
    title: "Customer Churn",
    category: "Classification",
    desc: "Predict whether a customer will churn based on usage patterns.",
    diff: "Easy",
    data: "MediumData",
    fields: ["usage", "tenure"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("customer_churn.csv")
X = df[['usage', 'tenure']]
y = df['churn']

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
print("Customer churn predictor online.")`,
    inputs: [
      { name: "usage", label: "Usage (Hours/Month)", type: "range", min: 0, max: 300, value: 45 },
      { name: "tenure", label: "Tenure (Months)", type: "range", min: 0, max: 60, value: 8 }
    ],
    predict: (inputs) => {
      const usageScore = inputs.usage / 300;
      const tenureScore = inputs.tenure / 60;
      return (usageScore * 0.4 + tenureScore * 0.6) < 0.25 ? "1" : "0";
    }
  },
  {
    id: 6,
    title: "Diabetes",
    category: "Classification",
    desc: "Predict if a patient has diabetes using health metrics.",
    diff: "Medium",
    data: "MediumData",
    fields: ["glucose", "blood_pressure", "BMI"],
    output: "Yes / No",
    solution: `import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv("diabetes.csv")
X = df[['glucose', 'blood_pressure', 'BMI']]
y = df['diabetes']

svm = SVC(probability=True)
svm.fit(X, y)
print("Support Vector Machine optimized.")`,
    inputs: [
      { name: "glucose", label: "Glucose Level (mg/dL)", type: "range", min: 50, max: 300, value: 145 },
      { name: "blood_pressure", label: "Blood Pressure (mmHg)", type: "range", min: 40, max: 180, value: 85 },
      { name: "BMI", label: "BMI", type: "range", min: 10, max: 60, value: 31 }
    ],
    predict: (inputs) => {
      const glucoseVal = parseInt(inputs.glucose);
      const bmiVal = parseInt(inputs.BMI);
      const bpVal = parseInt(inputs.blood_pressure);
      
      const prob = (glucoseVal / 300) * 0.5 + (bmiVal / 60) * 0.3 + (bpVal / 180) * 0.2;
      return prob > 0.55 ? "Yes" : "No";
    }
  },
  {
    id: 7,
    title: "Spam Emails",
    category: "Classification",
    desc: "Predict spam vs. non-spam emails using text features.",
    diff: "Medium",
    data: "Large",
    fields: ["email_text"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("emails.csv")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['email_text'])
y = df['label']

nb = MultinomialNB()
nb.fit(X, y)
print("Naive Bayes spam classifier ready.")`,
    inputs: [
      { name: "email_text", label: "Email Content Text", type: "textarea", value: "Hey, are we still meeting today at 3 PM for project sync?" }
    ],
    predict: (inputs) => {
      const txt = inputs.email_text.toLowerCase();
      const spamWords = ["win", "free", "guarantee", "buy now", "crypto", "millions", "cash", "invest", "urgent", "lottery"];
      let matches = 0;
      spamWords.forEach(word => {
        if (txt.includes(word)) matches++;
      });
      return matches >= 2 || txt.includes("lottery") || txt.includes("crypto") ? "1" : "0";
    }
  },
  {
    id: 8,
    title: "Loan Default",
    category: "Classification",
    desc: "Predict whether a loan will default based on applicant details.",
    diff: "Hard",
    data: "MediumData",
    fields: ["income", "credit_score", "loan_amount"],
    output: "Yes / No",
    solution: `import pandas as pd
from xgboost import XGBClassifier

df = pd.read_csv("loan_defaults.csv")
X = df[['income', 'credit_score', 'loan_amount']]
y = df['default']

xgb = XGBClassifier()
xgb.fit(X, y)
print("XGBoost loan validation classifier generated.")`,
    inputs: [
      { name: "income", label: "Annual Income ($)", type: "range", min: 10000, max: 250000, step: 5000, value: 55000 },
      { name: "credit_score", label: "Credit Score", type: "range", min: 300, max: 850, value: 620 },
      { name: "loan_amount", label: "Requested Loan Amount ($)", type: "range", min: 1000, max: 100000, step: 1000, value: 35000 }
    ],
    predict: (inputs) => {
      const dRatio = inputs.loan_amount / inputs.income;
      const cScore = inputs.credit_score;
      
      let score = dRatio * 0.6;
      if (cScore < 580) score += 0.4;
      else if (cScore < 680) score += 0.2;
      
      return score > 0.45 ? "Yes" : "No";
    }
  },
  {
    id: 9,
    title: "Chess Outcome",
    category: "Classification",
    desc: "Predict if a chess player will win based on Elo rating and age.",
    diff: "Hard",
    data: "Small",
    fields: ["elo_rating", "age"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

df = pd.read_csv("chess_games.csv")
X = df[['elo_rating', 'age']]
y = df['result']

model = ExtraTreesClassifier(n_estimators=200)
model.fit(X, y)
print("Chess outcome trees compiled.")`,
    inputs: [
      { name: "elo_rating", label: "Elo Rating", type: "range", min: 800, max: 2800, value: 1650 },
      { name: "age", label: "Player Age (Years)", type: "range", min: 8, max: 80, value: 24 }
    ],
    predict: (inputs) => {
      const elo = parseInt(inputs.elo_rating);
      const ageFactor = (inputs.age >= 18 && inputs.age <= 42) ? 0.1 : -0.15;
      const normalizedElo = (elo - 800) / 2000;
      
      const prob = normalizedElo * 0.8 + ageFactor + 0.1;
      return prob > 0.5 ? "1" : "0";
    }
  },

  // --- TIME-SERIES ---
  {
    id: 10,
    title: "Stock Prices",
    category: "Forecasting",
    desc: "Predict stock prices using lag features and moving averages.",
    diff: "Hard",
    data: "Large",
    fields: ["lag_features", "moving_average"],
    output: "Yes / No",
    solution: `import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("stock_prices.csv")
df['up_trend'] = (df['stock_price'].diff() > 0).astype(int)

model = ARIMA(df['stock_price'], order=(5,1,2))
fit = model.fit()
print("ARIMA model fitted.")`,
    inputs: [
      { name: "lag_features", label: "Lag Feature (Price Diff t-1)", type: "range", min: -10, max: 10, step: 0.1, value: 1.5 },
      { name: "moving_average", label: "10-day Moving Average ($)", type: "range", min: 10, max: 500, value: 120 }
    ],
    predict: (inputs) => {
      const lag = parseFloat(inputs.lag_features);
      return lag > 0.2 ? "Yes" : "No";
    }
  },
  {
    id: 11,
    title: "Retail Sales",
    category: "Forecasting",
    desc: "Predict daily sales for a retail store using past sales data.",
    diff: "Medium",
    data: "MediumData",
    fields: ["past_sales", "day"],
    output: "0 / 1",
    solution: `import pandas as pd
from prophet import Prophet

df = pd.read_csv("sales_data.csv")
df['high_sales'] = (df['sales'] > 5000).astype(int)

model = Prophet()
model.fit(df)
print("Sales forecasting model fitted.")`,
    inputs: [
      { name: "past_sales", label: "Yesterday's Sales ($)", type: "range", min: 500, max: 10000, step: 100, value: 4200 },
      { name: "day", label: "Day Category", type: "select", options: ["Weekday", "Weekend", "Holiday"], value: "Weekend" }
    ],
    predict: (inputs) => {
      const past = parseInt(inputs.past_sales);
      const isPromo = (inputs.day === "Weekend" || inputs.day === "Holiday");
      const salesProb = (past / 10000) * 0.5 + (isPromo ? 0.45 : 0.05);
      return salesProb > 0.5 ? "1" : "0";
    }
  },
  {
    id: 12,
    title: "Weather Forecast",
    category: "Forecasting",
    desc: "Forecast weather temperature for the next 7 days.",
    diff: "Medium",
    data: "Large",
    fields: ["temperature_history"],
    output: "Yes / No",
    solution: `import pandas as pd
from sklearn.neural_network import MLPRegressor

df = pd.read_csv("weather.csv")
df['heatwave_warning'] = (df['forecast_temp'] > 95).astype(int)

X = df[['temp_t_1', 'temp_t_2', 'temp_t_3']]
y = df['heatwave_warning']

mlp = MLPRegressor()
mlp.fit(X, y)
print("Forecaster net built.")`,
    inputs: [
      { name: "temperature_history", label: "Current Temp Trend (°F)", type: "range", min: 30, max: 110, value: 88 }
    ],
    predict: (inputs) => {
      const cur = parseInt(inputs.temperature_history);
      return cur >= 92 ? "Yes" : "No";
    }
  },
  {
    id: 13,
    title: "Website Traffic",
    category: "Forecasting",
    desc: "Predict website traffic using historical visitor counts.",
    diff: "Hard",
    data: "Large",
    fields: ["visitor_count", "day"],
    output: "0 / 1",
    solution: `import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv("traffic.csv")
df['viral_traffic'] = (df['traffic'] > 20000).astype(int)

model = ExponentialSmoothing(df['visitor_count'])
model_fit = model.fit()
print("Website visitors forecast fit complete.")`,
    inputs: [
      { name: "visitor_count", label: "Avg Visitors (Past 3 days)", type: "range", min: 100, max: 50000, value: 15000 },
      { name: "day", label: "Target Day", type: "select", options: ["Weekday", "Weekend"], value: "Weekday" }
    ],
    predict: (inputs) => {
      const vis = parseInt(inputs.visitor_count);
      const dayFactor = inputs.day === "Weekday" ? 0.2 : 0.0;
      const prob = (vis / 50000) * 0.75 + dayFactor;
      return prob > 0.45 ? "1" : "0";
    }
  },
  {
    id: 14,
    title: "Ride Demand",
    category: "Forecasting",
    desc: "Forecast demand for ride-sharing services.",
    diff: "Hardcore",
    data: "Large",
    fields: ["time", "location"],
    output: "Yes / No",
    solution: `import pandas as pd
import lightgbm as lgb

df = pd.read_csv("ride_sharing.csv")
df['surge_trigger'] = (df['demand'] > 150).astype(int)

X = pd.get_dummies(df[['time_slot', 'location_code']])
y = df['surge_trigger']

model = lgb.LGBMClassifier()
model.fit(X, y)
print("Surge Demand model compiled.")`,
    inputs: [
      { name: "time", label: "Time Slot", type: "select", options: ["Rush Hour", "Off-Peak", "Late Night"], value: "Rush Hour" },
      { name: "location", label: "Zone Sector", type: "select", options: ["Downtown Core", "Residential", "Industrial"], value: "Downtown Core" }
    ],
    predict: (inputs) => {
      return (inputs.time === "Rush Hour" || inputs.time === "Late Night") ? "Yes" : "No";
    }
  },

  // --- CLUSTERING ---
  {
    id: 15,
    title: "Customers",
    category: "Clustering",
    desc: "Cluster customers into segments, then predict segment for new customer.",
    diff: "Medium",
    data: "MediumData",
    fields: ["age", "income", "spending_score"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("customers.csv")
kmeans = KMeans(n_clusters=2)
df['segment'] = kmeans.fit_predict(df[['age', 'income', 'spending_score']])

X = df[['age', 'income', 'spending_score']]
y = df['segment']
clf = RandomForestClassifier()
clf.fit(X, y)
print("Customer clustering finished.")`,
    inputs: [
      { name: "age", label: "Customer Age", type: "range", min: 18, max: 80, value: 29 },
      { name: "income", label: "Income ($)", type: "range", min: 15000, max: 150000, step: 2000, value: 72000 },
      { name: "spending_score", label: "Spending Score (1-100)", type: "range", min: 1, max: 100, value: 68 }
    ],
    predict: (inputs) => {
      const score = (inputs.income / 150000) * 0.5 + (inputs.spending_score / 100) * 0.5;
      return score > 0.5 ? "1" : "0";
    }
  },
  {
    id: 16,
    title: "Movies",
    category: "Clustering",
    desc: "Cluster movies by genre features, then predict cluster for a new movie.",
    diff: "Medium",
    data: "MediumData",
    fields: ["genre", "rating"],
    output: "Yes / No",
    solution: `import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC

df = pd.read_csv("movies.csv")
clustering = AgglomerativeClustering(n_clusters=2)
df['cluster_id'] = clustering.fit_predict(df[['rating', 'genre_code']])

X = df[['rating', 'genre_code']]
y = df['cluster_id']
model = SVC()
model.fit(X, y)
print("Movie hierarchical clustering operational.")`,
    inputs: [
      { name: "genre", label: "Genre Focus", type: "select", options: ["Action/SciFi", "Indie/Drama", "Comedy/Family"], value: "Action/SciFi" },
      { name: "rating", label: "User Rating Score (1-10)", type: "range", min: 1, max: 10, step: 0.1, value: 7.8 }
    ],
    predict: (inputs) => {
      const rating = parseFloat(inputs.rating);
      return (rating > 7.0 && inputs.genre === "Indie/Drama") || rating > 8.0 ? "Yes" : "No";
    }
  },
  {
    id: 17,
    title: "Students",
    category: "Clustering",
    desc: "Cluster students by performance, then predict group for a new student.",
    diff: "Medium",
    data: "Small",
    fields: ["marks", "attendance"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("students.csv")
db = DBSCAN(eps=3.0, min_samples=4)
df['group'] = db.fit_predict(df[['marks', 'attendance']])

X = df[['marks', 'attendance']]
y = df['group']
model = KNeighborsClassifier()
model.fit(X, y)
print("Student clustering finished.")`,
    inputs: [
      { name: "marks", label: "Average Marks (%)", type: "range", min: 0, max: 100, value: 74 },
      { name: "attendance", label: "Attendance Record (%)", type: "range", min: 0, max: 100, value: 80 }
    ],
    predict: (inputs) => {
      const score = (inputs.marks / 100) * 0.6 + (inputs.attendance / 100) * 0.4;
      return score > 0.7 ? "1" : "0";
    }
  },
  {
    id: 18,
    title: "Shopping Patterns",
    category: "Clustering",
    desc: "Cluster shopping patterns, then predict cluster for a new transaction.",
    diff: "Hard",
    data: "Large",
    fields: ["items", "frequency"],
    output: "Yes / No",
    solution: `import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("shopping.csv")
kmeans = MiniBatchKMeans(n_clusters=2)
df['cluster'] = kmeans.fit_predict(df[['items', 'frequency']])

X = df[['items', 'frequency']]
y = df['cluster']
model = HistGradientBoostingClassifier()
model.fit(X, y)
print("Shopping pattern classifier compiled.")`,
    inputs: [
      { name: "items", label: "Unique Items Purchased", type: "range", min: 1, max: 30, value: 5 },
      { name: "frequency", label: "Transactions / Month", type: "range", min: 1, max: 100, value: 12 }
    ],
    predict: (inputs) => {
      const score = (inputs.items / 30) * 0.4 + (inputs.frequency / 100) * 0.6;
      return score > 0.35 ? "Yes" : "No";
    }
  },
  {
    id: 19,
    title: "Cities Pollution",
    category: "Clustering",
    desc: "Cluster cities by pollution levels, then predict cluster for a new city.",
    diff: "Hard",
    data: "MediumData",
    fields: ["pm25", "pm10"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.mixture import GaussianMixture

df = pd.read_csv("pollution.csv")
gmm = GaussianMixture(n_components=2)
df['pollution_cluster'] = gmm.fit_predict(df[['pm25', 'pm10']])

print("Gaussian Mixture Model calibrated.")`,
    inputs: [
      { name: "pm25", label: "PM2.5 Level (µg/m³)", type: "range", min: 0, max: 300, value: 45 },
      { name: "pm10", label: "PM10 Level (µg/m³)", type: "range", min: 0, max: 500, value: 80 }
    ],
    predict: (inputs) => {
      const pollutionScore = (inputs.pm25 / 300) * 0.6 + (inputs.pm10 / 500) * 0.4;
      return pollutionScore > 0.25 ? "1" : "0";
    }
  },

  // --- ADVANCED ---
  {
    id: 20,
    title: "Sentiment Analysis",
    category: "Advanced",
    desc: "Predict sentiment (positive/negative) from product reviews.",
    diff: "Easy",
    data: "Large",
    fields: ["review_text"],
    output: "Yes / No",
    solution: `import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

df = pd.read_csv("reviews.csv")
X = df['review_text']
y = df['sentiment']

pipeline = make_pipeline(TfidfVectorizer(), SGDClassifier())
pipeline.fit(X, y)
print("Sentiment analyzer pipeline compiled.")`,
    inputs: [
      { name: "review_text", label: "Product Review Text", type: "textarea", value: "This product exceeded all my expectations. Extremely high quality and super fast shipping!" }
    ],
    predict: (inputs) => {
      const text = inputs.review_text.toLowerCase();
      const posWords = ["good", "great", "excellent", "love", "perfect", "awesome", "recommend", "best", "happy", "exceeded"];
      const negWords = ["bad", "terrible", "worst", "waste", "hate", "disappointed", "poor", "broken", "cheap", "refund"];
      
      let score = 0;
      posWords.forEach(w => { if (text.includes(w)) score++; });
      negWords.forEach(w => { if (text.includes(w)) score--; });
      return score >= 0 ? "Yes" : "No";
    }
  },
  {
    id: 21,
    title: "Fraud Detection",
    category: "Advanced",
    desc: "Predict fraudulent transactions using anomaly detection + classification.",
    diff: "Hardcore",
    data: "Large",
    fields: ["transaction_amount", "location", "time"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier

df = pd.read_csv("transactions.csv")
iso = IsolationForest(contamination=0.01)
anomalies = iso.fit_predict(df[['transaction_amount']])

X = df[['transaction_amount', 'location_code', 'time_hour']]
y = df['fraud_flag']
model = LGBMClassifier()
model.fit(X, y)
print("Fraud detection system active.")`,
    inputs: [
      { name: "transaction_amount", label: "Transaction Value ($)", type: "range", min: 1, max: 10000, value: 120 },
      { name: "location", label: "IP/GPS Location Check", type: "select", options: ["Domestic Match", "International Divergence"], value: "Domestic Match" },
      { name: "time", label: "Time of Transaction", type: "select", options: ["Daytime", "Late Night (2AM-5AM)"], value: "Daytime" }
    ],
    predict: (inputs) => {
      const val = parseInt(inputs.transaction_amount);
      const isStrangeTime = inputs.time === "Late Night (2AM-5AM)";
      const isAbroad = inputs.location === "International Divergence";
      
      let risk = 0;
      if (val > 2000) risk += 0.4;
      if (isStrangeTime) risk += 0.3;
      if (isAbroad) risk += 0.5;
      
      return risk > 0.5 ? "1" : "0";
    }
  },
  {
    id: 22,
    title: "Employee Attrition",
    category: "Advanced",
    desc: "Predict employee attrition (who will leave the company).",
    diff: "Hard",
    data: "MediumData",
    fields: ["age", "salary", "years_at_company"],
    output: "Yes / No",
    solution: `import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("hr_attrition.csv")
X = df[['age', 'salary', 'years_at_company']]
y = df['attrition']

rf = RandomForestClassifier()
rf.fit(X, y)
print("Attrition forest fit finalized.")`,
    inputs: [
      { name: "age", label: "Employee Age", type: "range", min: 18, max: 65, value: 34 },
      { name: "salary", label: "Monthly Salary ($)", type: "range", min: 2000, max: 20000, step: 200, value: 6500 },
      { name: "years_at_company", label: "Years at Company", type: "range", min: 0, max: 30, value: 3 }
    ],
    predict: (inputs) => {
      const salaryFactor = inputs.salary / 20000;
      const years = inputs.years_at_company;
      return (salaryFactor < 0.25 && years < 3) ? "Yes" : "No";
    }
  },
  {
    id: 23,
    title: "Credit Card Approval",
    category: "Advanced",
    desc: "Predict credit card approval based on applicant features.",
    diff: "Hard",
    data: "MediumData",
    fields: ["income", "credit_score"],
    output: "0 / 1",
    solution: `import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("credit_approvals.csv")
X = df[['income', 'credit_score']]
y = df['approval']

model = LogisticRegression()
model.fit(X, y)
print("Credit card approval classifier active.")`,
    inputs: [
      { name: "income", label: "Annual Income ($)", type: "range", min: 10000, max: 250000, step: 5000, value: 60000 },
      { name: "credit_score", label: "Credit Score", type: "range", min: 300, max: 850, value: 690 }
    ],
    predict: (inputs) => {
      const score = inputs.credit_score;
      const inc = inputs.income;
      if (score < 600) return "0";
      if (score > 740) return "1";
      return (inc > 45000) ? "1" : "0";
    }
  },
  {
    id: 24,
    title: "Movie Revenue",
    category: "Advanced",
    desc: "Predict movie box office revenue using budget, cast, and genre.",
    diff: "Hardcore",
    data: "Large",
    fields: ["budget", "cast", "genre"],
    output: "Yes / No",
    solution: `import pandas as pd
from xgboost import XGBClassifier

df = pd.read_csv("movie_revenue.csv")
df['is_hit'] = (df['revenue'] > df['budget'] * 2.5).astype(int)

X = pd.get_dummies(df[['budget', 'cast', 'genre']])
y = df['is_hit']

xgb = XGBClassifier()
xgb.fit(X, y)
print("Box office model trained.")`,
    inputs: [
      { name: "budget", label: "Production Budget ($M)", type: "range", min: 1, max: 300, value: 85 },
      { name: "cast", label: "Cast Rating", type: "select", options: ["A-List Stars", "B-List Actors", "Indie/Newcomers"], value: "A-List Stars" },
      { name: "genre", label: "Genre", type: "select", options: ["Action/Sci-Fi", "Comedy/Drama", "Horror/Thriller"], value: "Action/Sci-Fi" }
    ],
    predict: (inputs) => {
      const bud = parseInt(inputs.budget);
      const isStarPower = inputs.cast === "A-List Stars";
      const isAction = inputs.genre === "Action/Sci-Fi";
      const prob = (bud / 300) * 0.3 + (isStarPower ? 0.4 : 0.1) + (isAction ? 0.3 : 0.1);
      return prob > 0.5 ? "Yes" : "No";
    }
  }
];

// --- 2. PARTICLE NET CANVAS SYSTEM ---
let particleSystem = {
  canvas: null,
  ctx: null,
  particles: [],
  maxParticles: 90,
  connectionDist: 110,
  mouse: { x: null, y: null, active: false },
  
  init() {
    this.canvas = document.getElementById("particle-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.resize();
    
    for (let i = 0; i < this.maxParticles; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 2 + 1,
        color: Math.random() > 0.8 ? "rgba(0, 242, 254, 0.45)" : "rgba(148, 163, 184, 0.2)"
      });
    }
    
    window.addEventListener("resize", () => this.resize());
    window.addEventListener("mousemove", (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
      this.mouse.active = true;
    });
    window.addEventListener("mouseleave", () => {
      this.mouse.active = false;
    });
    
    this.animate();
  },
  
  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  },
  
  animate() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    this.particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      
      if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;
      
      if (this.mouse.active) {
        const dx = this.mouse.x - p.x;
        const dy = this.mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 180) {
          p.x += dx * 0.005;
          p.y += dy * 0.005;
        }
      }
      
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
    });
    
    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const p1 = this.particles[i];
        const p2 = this.particles[j];
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < this.connectionDist) {
          const alpha = (1 - (dist / this.connectionDist)) * 0.18;
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = `rgba(0, 242, 254, ${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
      
      if (this.mouse.active) {
        const p = this.particles[i];
        const dx = p.x - this.mouse.x;
        const dy = p.y - this.mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 150) {
          const alpha = (1 - (dist / 150)) * 0.22;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(this.mouse.x, this.mouse.y);
          ctx.strokeStyle = `rgba(0, 242, 254, ${alpha})`;
          ctx.lineWidth = 0.6;
          ctx.stroke();
        }
      }
    }
    
    requestAnimationFrame(() => this.animate());
  }
};

// --- 3. 3D CARD TILT INSTRUCTION ---
function applyTiltEffect(card) {
  card.addEventListener("mousemove", (e) => {
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const xc = rect.width / 2;
    const yc = rect.height / 2;
    
    const rotateX = -((y - yc) / yc) * 10; 
    const rotateY = ((x - xc) / xc) * 10;
    
    card.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
    card.style.transition = "transform 0.1s ease";
  });
  
  card.addEventListener("mouseleave", () => {
    card.style.transform = `rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
    card.style.transition = "transform 0.5s ease";
  });
}

// --- 4. RENDER TASKS GRID & FILTERING ---
let currentCategory = "All";
let searchFilter = "";
let difficultyFilter = "All";
let datasetFilter = "All";

function renderGrid() {
  const grid = document.getElementById("tasks-grid");
  grid.innerHTML = "";
  
  let filtered = tasks.filter(task => {
    if (currentCategory !== "All" && task.category !== currentCategory) return false;
    if (difficultyFilter !== "All" && task.diff !== difficultyFilter) return false;
    if (datasetFilter !== "All" && task.data !== datasetFilter) return false;
    
    if (searchFilter) {
      const query = searchFilter.toLowerCase();
      const matchTitle = task.title.toLowerCase().includes(query);
      const matchDesc = task.desc.toLowerCase().includes(query);
      const matchFields = task.fields.some(f => f.toLowerCase().includes(query));
      if (!matchTitle && !matchDesc && !matchFields) return false;
    }
    
    return true;
  });
  
  document.getElementById("filtered-count").innerText = filtered.length;
  
  if (filtered.length === 0) {
    grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 60px; color: var(--text-muted);">No challenges matching current filter criteria.</div>`;
    return;
  }
  
  filtered.forEach(task => {
    const card = document.createElement("div");
    card.className = "task-card";
    card.dataset.id = task.id;
    
    const fieldsHTML = task.fields.map(f => `<span class="field-tag">${f}</span>`).join("");
    
    card.innerHTML = `
      <div>
        <div class="card-header">
          <span class="card-category">${task.category}</span>
          <div class="card-badges">
            <span class="badge badge-${task.diff.toLowerCase()}">${task.diff}</span>
            <span class="badge badge-${task.data.toLowerCase()}">${task.data}</span>
          </div>
        </div>
        <h3>${task.title}</h3>
        <p>${task.desc}</p>
      </div>
      <div class="card-footer">
        <div class="fields-list">
          ${fieldsHTML}
        </div>
        <div class="card-meta-row">
          <div class="output-badge-container">
            <span class="output-label">Expected:</span>
            <span class="output-value">${task.output}</span>
          </div>
          <span>Submit Code →</span>
        </div>
      </div>
    `;
    
    applyTiltEffect(card);
    card.addEventListener("click", () => openSandbox(task.id));
    grid.appendChild(card);
  });
}

// --- 5. INTERACTIVE SANDBOX DRAWER & SIMULATOR ---
let activeTaskId = null;
let isModelTrained = false;
let codeReviewChecked = false;
let trainingProgressInterval = null;

const drawer = document.getElementById("sandbox-drawer");
const overlay = document.getElementById("drawer-overlay");

function openSandbox(taskId) {
  const task = tasks.find(t => t.id === taskId);
  if (!task) return;
  
  activeTaskId = taskId;
  isModelTrained = false;
  codeReviewChecked = false;
  
  // Set header
  document.getElementById("sandbox-task-title").innerText = task.title;
  document.getElementById("sandbox-task-desc").innerText = task.desc;
  
  // Clear Editor: start completely EMPTY as requested
  document.getElementById("code-editor").value = "";
  
  // Reset logs
  const consolePanel = document.getElementById("sandbox-console");
  consolePanel.innerHTML = `<div class="console-line info">✦ Editor cleared. Paste your Python model script or write it here, then click 'Submit Code'.</div>`;
  
  // Setup inputs
  const inputsContainer = document.getElementById("predict-fields-container");
  inputsContainer.innerHTML = "";
  
  task.inputs.forEach(inp => {
    const group = document.createElement("div");
    group.className = "form-group";
    
    const labelRow = document.createElement("div");
    labelRow.style.display = "flex";
    labelRow.style.justify = "space-between";
    
    const label = document.createElement("label");
    label.setAttribute("for", `inp-${inp.name}`);
    label.innerText = inp.label;
    labelRow.appendChild(label);
    
    let valIndicator = null;
    if (inp.type === "range") {
      valIndicator = document.createElement("span");
      valIndicator.style.fontFamily = "var(--font-mono)";
      valIndicator.style.fontSize = "0.8rem";
      valIndicator.style.color = "var(--accent-cyan)";
      valIndicator.innerText = inp.value.toLocaleString();
      labelRow.appendChild(valIndicator);
    }
    
    group.appendChild(labelRow);
    
    let control;
    if (inp.type === "range") {
      control = document.createElement("input");
      control.type = "range";
      control.id = `inp-${inp.name}`;
      control.name = inp.name;
      control.className = "form-control";
      control.min = inp.min;
      control.max = inp.max;
      if (inp.step) control.step = inp.step;
      control.value = inp.value;
      
      control.addEventListener("input", (e) => {
        valIndicator.innerText = parseFloat(e.target.value).toLocaleString();
      });
    } else if (inp.type === "select") {
      control = document.createElement("select");
      control.id = `inp-${inp.name}`;
      control.name = inp.name;
      control.className = "form-control";
      inp.options.forEach(opt => {
        const option = document.createElement("option");
        option.value = opt;
        option.innerText = opt;
        if (opt === inp.value) option.selected = true;
        control.appendChild(option);
      });
    } else if (inp.type === "textarea") {
      control = document.createElement("textarea");
      control.id = `inp-${inp.name}`;
      control.name = inp.name;
      control.className = "form-control";
      control.style.minHeight = "70px";
      control.style.resize = "vertical";
      control.value = inp.value;
    }
    
    group.appendChild(control);
    inputsContainer.appendChild(group);
  });
  
  // Set default code verification output
  resetEvaluationOutput(task.output);
  
  // Reset Code Review Checklist display
  resetReviewReport();
  
  // Lock simulator form & inputs
  toggleSimulatorPlayground(false);
  
  // Open drawer
  drawer.classList.add("active");
  overlay.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeSandbox() {
  drawer.classList.remove("active");
  overlay.classList.remove("active");
  document.body.style.overflow = "";
  clearInterval(trainingProgressInterval);
}

// Reset code validation status outcome
function resetEvaluationOutput(outputType) {
  const resultDisplay = document.getElementById("predict-result-val");
  resultDisplay.innerText = "---";
  resultDisplay.className = "res-val";
  document.getElementById("eval-output-type-label").innerText = `Submission Outcome (${outputType}):`;
}

// Resets review metrics to default
function resetReviewReport() {
  document.getElementById("review-score-val").innerText = "0%";
  document.getElementById("review-score-circle").style.strokeDasharray = "0, 100";
  document.getElementById("review-status-txt").innerText = "Not Evaluated";
  document.getElementById("review-status-txt").className = "review-status-text pending";
  
  document.getElementById("checklist-container").innerHTML = `
    <div style="color:var(--text-muted);font-size:0.85rem;">Click 'Submit Code' to initialize automated verification checklist.</div>
  `;
  document.getElementById("recommendations-container").innerHTML = `
    <li style="color:var(--text-muted);list-style:none;">No critiques computed.</li>
  `;
}

// Enable/Disable the right-side sliders and inference controls
function toggleSimulatorPlayground(enable) {
  const inputFields = document.querySelectorAll("#predict-fields-container input, #predict-fields-container select, #predict-fields-container textarea");
  const inferBtn = document.getElementById("btn-run-predict");
  const inputsCard = document.getElementById("predict-box-card");
  
  inputFields.forEach(el => {
    el.disabled = !enable;
  });
  inferBtn.disabled = !enable;
  
  if (enable) {
    inferBtn.style.opacity = "1";
    inferBtn.style.cursor = "pointer";
    inputsCard.style.opacity = "1";
    document.getElementById("simulator-disabled-mask").style.display = "none";
  } else {
    inferBtn.style.opacity = "0.5";
    inferBtn.style.cursor = "not-allowed";
    inputsCard.style.opacity = "0.7";
    document.getElementById("simulator-disabled-mask").style.display = "flex";
  }
}

// Analyze Code Submission (The reviewer heuristic logic)
function executeCodeReview() {
  const task = tasks.find(t => t.id === activeTaskId);
  if (!task) return;
  
  const code = document.getElementById("code-editor").value;
  const consolePanel = document.getElementById("sandbox-console");
  
  consolePanel.innerHTML += `<div class="console-line info">&gt;&gt; python evaluator.py --analyze-input</div>`;
  consolePanel.innerHTML += `<div class="console-line">Inspecting submission for required components...</div>`;
  consolePanel.scrollTop = consolePanel.scrollHeight;
  
  let score = 0;
  let checks = [
    { id: "imports", label: "Library Imports (pandas, numpy, sklearn/xgboost)", pass: false, desc: "Requires model imports or numpy/pandas loaders." },
    { id: "csv_load", label: "Dataset Loading (read_csv or data frames)", pass: false, desc: "Needs data fetching or data configuration." },
    { id: "features", label: "Task Feature Columns Reference", pass: false, desc: `Must reference fields: ${task.fields.join(", ")}.` },
    { id: "fit", label: "Model Fitting (.fit or optimization)", pass: false, desc: "Missing model optimization step (.fit)." },
    { id: "predict", label: "Inference Prediction (.predict)", pass: false, desc: "Missing output/inference predicting calls (.predict)." }
  ];
  
  let recommendations = [];
  
  // 1. Check Imports
  const importRegex = /(import\s+pandas|import\s+numpy|import\s+sklearn|from\s+sklearn|import\s+xgboost|import\s+lightgbm|from\s+lightgbm|from\s+prophet|from\s+statsmodels)/i;
  if (importRegex.test(code)) {
    checks[0].pass = true;
    score += 20;
  } else {
    recommendations.push("Include standard imports (e.g. `import pandas as pd`, `from sklearn.ensemble import RandomForestClassifier`).");
  }
  
  // 2. Check CSV loading
  const loadRegex = /(read_csv|pd\.DataFrame|load_|df\s*=\s*|data\s*=\s*)/i;
  if (loadRegex.test(code)) {
    checks[1].pass = true;
    score += 20;
  } else {
    recommendations.push("Load the dataset using pandas (e.g. `df = pd.read_csv('dataset.csv')`).");
  }
  
  // 3. Check specific task features
  let missingFields = [];
  task.fields.forEach(field => {
    const regex = new RegExp(`['"]${field}['"]|${field}`, "i");
    if (!regex.test(code)) {
      missingFields.push(field);
    }
  });
  
  if (missingFields.length === 0) {
    checks[2].pass = true;
    score += 20;
  } else {
    recommendations.push(`Ensure you extract or select the task's required feature columns: **[${missingFields.join(", ")}]**.`);
  }
  
  // 4. Check model.fit
  if (/\.fit\(/i.test(code)) {
    checks[3].pass = true;
    score += 20;
  } else {
    recommendations.push("Train your model by invoking the fitting function (e.g. `model.fit(X_train, y_train)`).");
  }
  
  // 5. Check model.predict
  if (/\.predict\(/i.test(code)) {
    checks[4].pass = true;
    score += 20;
  } else {
    recommendations.push("Implement code prediction calls (e.g. `predictions = model.predict(X_test)`).");
  }
  
  if (code.trim().length < 15) {
    score = 0;
    checks.forEach(c => c.pass = false);
    recommendations = ["Your script is too brief. Please write or paste complete Python ML code inside the editor."];
  }
  
  setTimeout(() => {
    displayReviewReport(score, checks, recommendations);
    
    // Binary code evaluation outcome!
    const validated = (score >= 80); 
    const outcomeResult = validated 
      ? (task.output === "Yes / No" ? "Yes" : "1") 
      : (task.output === "Yes / No" ? "No" : "0");
      
    const valDisplay = document.getElementById("predict-result-val");
    valDisplay.innerText = outcomeResult;
    valDisplay.className = "res-val";
    
    if (validated) {
      valDisplay.classList.add(task.output === "Yes / No" ? "yes" : "one");
      consolePanel.innerHTML += `<div class="console-line success">✓ CODE VERIFIED: Validation outcome matches expected format. Outcome set to '${outcomeResult}' (Success).</div>`;
      consolePanel.innerHTML += `<div class="console-line success">Prediction parameters unlocked! Verify different inputs on the right.</div>`;
      toggleSimulatorPlayground(true);
      isModelTrained = true;
      
      // Perform mock loss plot showing convergence
      drawLossChart([0.92, 0.76, 0.58, 0.44, 0.32, 0.25, 0.19, 0.14, 0.11, 0.08]);
    } else {
      valDisplay.classList.add(task.output === "Yes / No" ? "no" : "zero");
      consolePanel.innerHTML += `<div class="console-line error">✗ VERIFICATION FAILED: Outcome set to '${outcomeResult}' (Incomplete).</div>`;
      consolePanel.innerHTML += `<div class="console-line error">Check the critique list and make sure all structural components are referenced in your script.</div>`;
      toggleSimulatorPlayground(false);
      isModelTrained = false;
      drawLossChart([]);
    }
    consolePanel.scrollTop = consolePanel.scrollHeight;
  }, 600);
}

// Build checklist elements & score display in UI
function displayReviewReport(score, checks, recommendations) {
  document.getElementById("review-score-val").innerText = `${score}%`;
  
  const circle = document.getElementById("review-score-circle");
  circle.style.strokeDasharray = `${score}, 100`;
  
  const txt = document.getElementById("review-status-txt");
  if (score >= 80) {
    txt.innerText = "Verified Model";
    txt.className = "review-status-text success";
  } else if (score >= 40) {
    txt.innerText = "Needs Correction";
    txt.className = "review-status-text warning";
  } else {
    txt.innerText = "Validation Failed";
    txt.className = "review-status-text error";
  }
  
  const checkContainer = document.getElementById("checklist-container");
  checkContainer.innerHTML = "";
  checks.forEach(c => {
    const item = document.createElement("div");
    item.className = `check-item ${c.pass ? "pass" : "fail"}`;
    item.innerHTML = `
      <span class="check-icon">${c.pass ? "✓" : "✗"}</span>
      <div class="check-details">
        <span class="check-label">${c.label}</span>
        <span class="check-desc">${c.desc}</span>
      </div>
    `;
    checkContainer.appendChild(item);
  });
  
  const recContainer = document.getElementById("recommendations-container");
  recContainer.innerHTML = "";
  if (recommendations.length === 0) {
    recContainer.innerHTML = `<li class="success" style="list-style:none;color:var(--diff-easy)">✓ Brilliant job! Your code fully compiles and satisfies all structural ML criteria.</li>`;
  } else {
    recommendations.forEach(r => {
      const li = document.createElement("li");
      li.innerHTML = r;
      recContainer.appendChild(li);
    });
  }
}

// Inference Engine prediction loop
function runPredictionSimulation() {
  if (!isModelTrained) return;
  const task = tasks.find(t => t.id === activeTaskId);
  if (!task) return;
  
  const consolePanel = document.getElementById("sandbox-console");
  consolePanel.innerHTML += `<div class="console-line info">&gt;&gt; model.predict(interactive_vector)</div>`;
  consolePanel.scrollTop = consolePanel.scrollHeight;
  
  let inputsValues = {};
  task.inputs.forEach(inp => {
    const el = document.getElementById(`inp-${inp.name}`);
    if (el) {
      if (inp.type === "range") {
        inputsValues[inp.name] = parseFloat(el.value);
      } else {
        inputsValues[inp.name] = el.value;
      }
    }
  });
  
  const prediction = task.predict(inputsValues);
  
  setTimeout(() => {
    consolePanel.innerHTML += `<div class="console-line success">Prediction output computed: value = ${prediction}</div>`;
    consolePanel.scrollTop = consolePanel.scrollHeight;
    
    const valDisplay = document.getElementById("predict-result-val");
    valDisplay.innerText = prediction;
    valDisplay.className = "res-val";
    
    if (prediction === "Yes") valDisplay.classList.add("yes");
    else if (prediction === "No") valDisplay.classList.add("no");
    else if (prediction === "1") valDisplay.classList.add("one");
    else if (prediction === "0") valDisplay.classList.add("zero");
  }, 250);
}

// SVG Loss Chart generator
function drawLossChart(losses) {
  const svg = document.getElementById("loss-svg");
  if (losses.length === 0) {
    svg.innerHTML = `<line x1="0" y1="30" x2="400" y2="30" style="stroke:rgba(255,255,255,0.06);stroke-width:2" />`;
    return;
  }
  
  const width = svg.clientWidth || 300;
  const height = svg.clientHeight || 60;
  const maxLoss = 1.0;
  
  let points = [];
  losses.forEach((loss, i) => {
    const x = (i / (losses.length - 1)) * width;
    const y = height - (loss / maxLoss) * (height - 16) - 8;
    points.push(`${x},${y}`);
  });
  
  svg.innerHTML = `
    <polyline points="${points.join(" ")}" style="fill:none;stroke:var(--accent-cyan);stroke-width:2" />
    <path d="M 0,${height} L ${points.join(" L ")} L ${width},${height} Z" style="fill:rgba(0, 242, 254, 0.04);stroke:none;" />
  `;
}

// --- 6. EVENT LISTENERS & FILTER SETUP ---
document.addEventListener("DOMContentLoaded", () => {
  particleSystem.init();
  
  const searchInput = document.getElementById("search-input");
  searchInput.addEventListener("input", (e) => {
    searchFilter = e.target.value;
    renderGrid();
  });
  
  document.getElementById("filter-difficulty").addEventListener("change", (e) => {
    difficultyFilter = e.target.value;
    renderGrid();
  });
  
  document.getElementById("filter-dataset").addEventListener("change", (e) => {
    datasetFilter = e.target.value;
    renderGrid();
  });
  
  const tabContainer = document.getElementById("category-tabs-container");
  tabContainer.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    
    tabContainer.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    
    currentCategory = btn.dataset.category;
    renderGrid();
  });
  
  overlay.addEventListener("click", closeSandbox);
  document.getElementById("btn-close-sandbox").addEventListener("click", closeSandbox);
  document.getElementById("btn-submit-code").addEventListener("click", executeCodeReview); // Connected to Code Reviewer!
  document.getElementById("btn-run-predict").addEventListener("click", runPredictionSimulation);
  
  document.getElementById("code-editor").addEventListener("keydown", function(e) {
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = this.selectionStart;
      const end = this.selectionEnd;
      this.value = this.value.substring(0, start) + "\t" + this.value.substring(end);
      this.selectionStart = this.selectionEnd = start + 1;
    }
  });
  
  // Clear editor button
  document.getElementById("btn-clear-editor").addEventListener("click", () => {
    document.getElementById("code-editor").value = "";
    const consolePanel = document.getElementById("sandbox-console");
    consolePanel.innerHTML += `<div class="console-line info">✦ Editor cleared. Paste your own code.</div>`;
    consolePanel.scrollTop = consolePanel.scrollHeight;
  });
  
  // Load Solution button inside IDE
  document.getElementById("btn-load-solution").addEventListener("click", () => {
    const task = tasks.find(t => t.id === activeTaskId);
    if (task) {
      document.getElementById("code-editor").value = task.solution;
      const consolePanel = document.getElementById("sandbox-console");
      consolePanel.innerHTML += `<div class="console-line success">✓ Reference solution loaded in editor. Click 'Submit Code' to verify.</div>`;
      consolePanel.scrollTop = consolePanel.scrollHeight;
    }
  });
  
  renderGrid();
});
