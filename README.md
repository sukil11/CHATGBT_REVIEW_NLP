# CHATGBT_REVIEW_NLP
***📁 Folder Structure***
bash
Copy
Edit
chatgpt-sentiment-analysis/
│
├── app/                         # Main app logic
│   └── app.py                   # Streamlit dashboard
│
├── model/                       # Model artifacts
│   ├── sentiment_model.h5       # Pre-trained LSTM model
│   └── tokenizer.json           # Tokenizer config (optional)
│
├── data/                        # Example dataset(s)
│   └── sample_reviews.csv
│
├── assets/                      # Static assets (images, etc.)
│   └── background.jpg
│
├── requirements.txt             # Project dependencies
├── README.md                    # Documentation & usage guide
├── .gitignore                   # Ignore files for Git
└── LICENSE                      # Open-source license
***✅ Project Deliverables***
🧠 sentiment_model.h5
Trained Keras model (LSTM)

Used to predict sentiment from review text

📄 app.py
Streamlit-based dashboard

Upload CSV → EDA + Sentiment prediction

Dynamic wordclouds, visual insights, model inference

***📊 sample_reviews.csv***
Contains columns like:

review, rating, date, platform, location, verified_purchase, version

***🧪 EDA Insights (via dropdown in app)***
Overall sentiment distribution

Sentiment by rating

Word clouds by sentiment

Time trends (if date column exists)

Verified vs unverified review sentiment

Review length vs sentiment

Location, platform, version-wise sentiment breakdown

Negative feedback themes

***💬 Realtime Prediction***
Enter a review → Instant sentiment result

***📦 requirements.txt***
nginx
Copy
Edit
streamlit
pandas
numpy
tensorflow
matplotlib
seaborn
wordcloud



