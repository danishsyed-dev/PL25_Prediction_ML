# Premier League Match Prediction System

A machine learning system that predicts Premier League football match outcomes using historical data and team statistics.

## Features
- Downloads real Premier League data from multiple seasons
- Calculates team strength, form, and performance metrics
- Uses Random Forest classifier for predictions
- Provides probability estimates for match outcomes

## Usage
```python
python main.py
```

## Requirements
- pandas
- scikit-learn
- requests
- numpy

## How It Works
1. Downloads Premier League data from football-data.co.uk
2. Calculates team features (strength, form, goals avg)
3. Trains Random Forest model on historical matches
4. Predicts outcomes for new matches

## Example Output
```
Prediction: Arsenal wins
Probabilities:
Manchester United wins: 35.2%
Draw: 28.1%
Arsenal wins: 36.7%
```
