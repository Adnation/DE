import numpy as np
from model import input_fn
from model import build_estimator
from datetime import datetime, timedelta
import pandas as pd


predict_df = pd.read_csv('raw_data.csv')

start = datetime.strptime('1970-01-01', '%Y-%m-%d')

# Get actual date from difference
# df['signup_date'] = df['signup_date'].apply(lambda x: start + timedelta(days=x))
predict_df['last_service_use_date'] = predict_df['last_service_use_date'].apply(lambda x: start + timedelta(days=x))

# df.rename(columns={'Unnamed: 0': 'user_id'}, inplace=True)

# Get user's recency
predict_df['recency'] = predict_df['last_service_use_date'].apply(lambda x: (predict_df.last_service_use_date.max() - x).days)

# Convert True False to 0 & 1
predict_df.loc[predict_df['business_service'] == True, 'business_service'] = '1'
predict_df.loc[predict_df['business_service'] == False, 'business_service'] = '0'

predict_df['is_retained'] = 0
# df.loc[df['last_service_use_date'].dt.month.isin([6,7]), 'is_retained'] = 1

predict_df.business_service = predict_df.business_service.astype(str)
predict_df.dropna(inplace=True)

m = build_estimator('model_dir')
predicted_values = list(m.predict(input_fn=lambda: input_fn(predict_df)))
probs = list(m.predict_proba(input_fn=lambda: input_fn(predict_df)))

predict_df['predicted_values'] = predicted_values
predict_df['probs'] = probs

predict_df.to_csv('predicttions.csv')
