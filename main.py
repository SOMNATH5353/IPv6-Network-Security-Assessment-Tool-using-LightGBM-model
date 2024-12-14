from models.model import train_gbm_model, predict_and_fill
from src.preprocessing import preprocess_data
from models.metrics import calculate_metrics, display_metrics

# Preprocess the data
preprocess_data('data/ipv6_data.csv')

# Train the model
train_gbm_model('data/preprocessed_data.csv')

# Predict and fill
predict_and_fill('data/sample.csv', 'models/gbm_model.pkl')
