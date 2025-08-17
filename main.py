import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import warnings
warnings.filterwarnings("ignore")

class PLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.match_data = None

    def get_football_data(self):
        print("Downloading Premier League data...")
        season_urls = {
            '2022-2023': 'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
            '2023-2024': 'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
            '2024-2025': 'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
        }
        all_matches = []

        for season, url in season_urls.items():
            try:
                print(f" Getting {season} season...")
                response = requests.get(url, timeout=10)
                from io import StringIO
                season_data = pd.read_csv(StringIO(response.text))
                clean_data = self.clean_season_data(season_data, season)
                all_matches.append(clean_data)
                print(f"Got {len(clean_data)} matches from {season}")
            
            except Exception as e:
                print(f"Couldn't get {season} data: {e}")

        if all_matches:
            self.match_data = pd.concat(all_matches, ignore_index=True)
            print(f"Total matches loaded: {len(self.match_data)}")
            return True
        else:
            print("No data could be loaded")
            return False
    def clean_season_data(self, data, season):
        matches = []

        for _, row in data.iterrows():
            try:
                if pd.isna(row.get('FTHG')) or pd.isna(row.get('FTAG')):
                    continue

                home_team = row.get('HomeTeam', '')
                away_team = row.get('AwayTeam', '')
                home_goals = int(row['FTHG'])
                away_goals = int(row['FTAG'])

                if home_goals > away_goals:
                    result = 'H'
                elif away_goals > home_goals:
                    result = 'A'
                else:
                    result = 'D'

                matches.append({
                    'season': season,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'result': result
                })
            
            except(ValueError, TypeError):
                continue

        return pd.DataFrame(matches)

    def calculate_simple_features(self, data):
        enhanced_data = data.copy().sort_values('season').reset_index(drop=True)

        enhanced_data['home_team_strength'] = 50
        enhanced_data['away_team_strength'] = 50
        enhanced_data['home_team_form'] = 5
        enhanced_data['away_team_form'] = 5
        enhanced_data['home_goals_avg'] = 1.5
        enhanced_data['away_goals_avg'] = 1.5
        enhanced_data['home_goals_conceded_avg'] = 1.5
        enhanced_data['away_goals_conceded_avg'] = 1.5
        enhanced_data['home_advantage'] = 1

        for i, match in enhanced_data.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            home_history = self.get_team_history(enhanced_data, home_team, i, games=5)
            away_history = self.get_team_history(enhanced_data, away_team, i, games=5)
            
            home_stats = self.calculate_team_stats(home_history, home_team)
            away_stats = self.calculate_team_stats(away_history, away_team)
            
            enhanced_data.loc[i, 'home_team_strength'] = home_stats['strength']
            enhanced_data.loc[i, 'away_team_strength'] = away_stats['strength']
            enhanced_data.loc[i, 'home_team_form'] = home_stats['form']
            enhanced_data.loc[i, 'away_team_form'] = away_stats['form']
            enhanced_data.loc[i, 'home_goals_avg'] = home_stats['goals_for']
            enhanced_data.loc[i, 'away_goals_avg'] = away_stats['goals_for']
            enhanced_data.loc[i, 'home_goals_conceded_avg'] = home_stats['goals_against']
            enhanced_data.loc[i, 'away_goals_conceded_avg'] = away_stats['goals_against']

        print(f"Features calculated for {len(enhanced_data)} matches")
        return enhanced_data
    
    def get_team_history(self, data, team, current_match_index, games=5):
        team_matches = data[
            ((data['home_team'] == team) | (data['away_team'] == team)) &
            (data.index < current_match_index)
        ]
        return team_matches.tail(games)
    
    def calculate_team_stats(self, history, team):
        if len(history) == 0:
            return {
                'strength': 50,
                'form': 5,
                'goals_for': 1.5,
                'goals_against': 1.5
            }
        
        points = 0
        goals_scored = 0
        goals_conceded = 0

        for _, match in history.iterrows():
            if match['home_team'] == team:
                goals_scored += match['home_goals']
                goals_conceded += match['away_goals']
                if match['result'] == 'H':
                    points += 3
                elif match['result'] == 'D':
                    points += 1
            else:
                goals_scored += match['away_goals']
                goals_conceded += match['home_goals']
                if match['result'] == 'A':
                    points += 3
                elif match['result'] == 'D':
                    points += 1

        num_games = len(history)

        goals_per_game = goals_scored / num_games
        goals_conceded_per_game = goals_conceded / num_games

        strength = (points / num_games) * 20 + 20

        return {
            'strength': min(90, max(10, strength)),
            'form': points,
            'goals_for': goals_per_game,
            'goals_against': goals_conceded_per_game
        }

    def prepare_training_data(self, data):
        print("Preparing data for training...")

        feature_columns = [
            'home_team_strength',
            'away_team_strength',
            'home_team_form',
            'away_team_form',
            'home_goals_avg',
        ]

        x = data[feature_columns].copy()
        y = data['result']

        print(f"Training data shape: {x.shape}")
        print(f"Features: {feature_columns}")

        return x, y
    
    def train_model(self):
        if self.match_data is None:
            print("No data loaded! Run get_football_data() first.")
            return False
        
        data_with_features = self.calculate_simple_features(self.match_data)
        training_data = data_with_features.iloc[50:].reset_index(drop=True)

        X, y = self.prepare_training_data(training_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=39
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"Accuracy: {accuracy:.1%} (on {len(X_test)} test matches)")

        feature_names = X.columns
        importances = self.model.feature_importances_

        print(f"Most important features:")
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{name}: {imp:.3f}")
        return True

    def predict_match(self, home_team, away_team):
        print(f"Predicting: {home_team} vs {away_team}")
        print("=" * 50)

        recent_matches = self.match_data.tail(100)

        home_recent = recent_matches[
            (recent_matches['home_team'] == home_team) |
            (recent_matches['away_team'] == home_team)
        ].tail(5)

        away_recent = recent_matches[
            (recent_matches['home_team'] == away_team) |
            (recent_matches['away_team'] == away_team)
        ].tail(5)

        home_stats = self.calculate_team_stats(home_recent, home_team)
        away_stats = self.calculate_team_stats(away_recent, away_team)

        match_features = pd.DataFrame({
            'home_team_strength': [home_stats['strength']],
            'away_team_strength': [away_stats['strength']],
            'home_team_form': [home_stats['form']],
            'away_team_form': [away_stats['form']],
            'home_goals_avg': [home_stats['goals_for']],
        })

        prediction = self.model.predict(match_features)[0]
        probabilities = self.model.predict_proba(match_features)[0]

        result = {'H': f"{home_team} wins", "D": "Draw", "A": f"{away_team} wins"}
        classes = self.model.classes_

        print(f"Prediction: {result[prediction]}")
        print(f"Probabilities:")
        for i, outcome in enumerate(classes):
            prob = probabilities[i]
            print(f"{result[outcome]}: {prob:.1%}")

        return prediction, probabilities
    
def main():
    print("Premier League Prediction Manchester United vs Arsenal")
    print("=" * 50)
    
    predictor = PLPredictor()
    if not predictor.get_football_data():
        print("Couldn't get data. Stopping here.")
        return
    if not predictor.train_model():
        print("Couldn't train model. Stopping here.")
        return
    
    print("\nMATCH PREDICTION FOR AUGUST 17, 2025")
    prediction, probabilities = predictor.predict_match("Manchester United", "Arsenal")
    return predictor

if __name__ == "__main__":
    model = main()