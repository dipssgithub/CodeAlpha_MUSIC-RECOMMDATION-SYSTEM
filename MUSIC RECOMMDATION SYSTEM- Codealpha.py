#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datetime

class SpotifyRecommendationSystem:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.user_features = {}
        self.song_features = {}
    
    def prepare_data(self, user_song_history):
        """
        Prepare data from user listening history
        
        Args:
            user_song_history: DataFrame with columns [user_id, song_id, timestamp, features...]
            
        Returns:
            X: Features matrix
            y: Target labels (1 for repeated listen within a month, 0 otherwise)
        """
        # Convert timestamp to datetime if it's not already
        if not isinstance(user_song_history['timestamp'].iloc[0], datetime.datetime):
            user_song_history['timestamp'] = user_song_history['timestamp'].apply(
                lambda x: datetime.datetime.fromtimestamp(x))
        
        # Add time-based features
        user_song_history['hour_of_day'] = user_song_history['timestamp'].apply(lambda x: x.hour)
        user_song_history['day_of_week'] = user_song_history['timestamp'].apply(lambda x: x.weekday())
        
        # Convert listen_date to string to avoid datetime aggregation issues
        user_song_history['listen_date'] = user_song_history['timestamp'].apply(
            lambda x: x.date().isoformat())
        
        # Create feature for whether song was listened to again within a month
        repeated_listens = []
        
        for (user, song), group in user_song_history.groupby(['user_id', 'song_id']):
            date_objects = [datetime.date.fromisoformat(d) for d in sorted(group['listen_date'].unique())]
            
            for i in range(len(date_objects) - 1):
                # Check if next listen is within 30 days
                if (date_objects[i+1] - date_objects[i]).days <= 30:
                    repeated_listens.append((user, song, date_objects[i].isoformat(), 1))
                else:
                    repeated_listens.append((user, song, date_objects[i].isoformat(), 0))
            
            # Last listen has no known future listens in our data
            repeated_listens.append((user, song, date_objects[-1].isoformat(), 0))
        
        repeated_df = pd.DataFrame(repeated_listens, 
                                  columns=['user_id', 'song_id', 'listen_date', 'repeated'])
        
        # Merge with original data
        merged_data = pd.merge(
            user_song_history,
            repeated_df,
            on=['user_id', 'song_id', 'listen_date']
        )
        
        # Extract features (example features - would use actual audio features in real system)
        feature_columns = ['hour_of_day', 'day_of_week', 'tempo', 'energy', 'danceability', 
                          'acousticness', 'instrumentalness', 'user_age', 'user_country']
        
        X = merged_data[feature_columns]
        y = merged_data['repeated']
        
        return X, y
    
    def train(self, user_song_history):
        """
        Train the recommendation model
        
        Args:
            user_song_history: DataFrame with user listening data
            
        Returns:
            Training metrics
        """
        X, y = self.prepare_data(user_song_history)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        # Build user and song feature representations for recommendations
        self._build_feature_representations(user_song_history)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _build_feature_representations(self, user_song_history):
        """Build user and song embeddings from history data"""
        # Exclude non-numeric columns that could cause aggregation issues
        numeric_cols = user_song_history.select_dtypes(include=['number']).columns
        
        # In a real system, this would build collaborative filtering features
        # Simplified version: average numeric features by user and song
        user_features = user_song_history.groupby('user_id')[numeric_cols].mean()
        song_features = user_song_history.groupby('song_id')[numeric_cols].mean()
        
        # Store in instance
        self.user_features = {user: user_features.loc[user].values 
                             for user in user_features.index}
        self.song_features = {song: song_features.loc[song].values 
                             for song in song_features.index}
    
    def predict_repeat_listen(self, user_id, song_id, timestamp, additional_features):
        """
        Predict if a user will repeatedly listen to a song
        
        Args:
            user_id: ID of the user
            song_id: ID of the song
            timestamp: Unix timestamp of the listen event
            additional_features: Dict of additional features
            
        Returns:
            Probability of repeated listen
        """
        # Convert timestamp to hour and day features
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour_of_day = dt.hour
        day_of_week = dt.weekday()
        
        # Combine features
        features = [hour_of_day, day_of_week]
        
        # Add other features from the additional_features dict
        for feature in ['tempo', 'energy', 'danceability', 
                        'acousticness', 'instrumentalness', 'user_age', 'user_country']:
            features.append(additional_features.get(feature, 0))
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict probability
        return self.model.predict_proba(features_scaled)[0][1]
    
    def recommend_songs(self, user_id, candidate_songs, top_n=10):
        """
        Recommend songs for a user from a list of candidates
        
        Args:
            user_id: ID of the user
            candidate_songs: List of song IDs to consider
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended song IDs
        """
        current_timestamp = datetime.datetime.now().timestamp()
        predictions = []
        
        # For each candidate song, predict likelihood of repeat listen
        for song_id in candidate_songs:
            # Create feature dict - in real system, would get actual song features
            if song_id in self.song_features:
                additional_features = {
                    'tempo': 120,  # Example values
                    'energy': 0.7,
                    'danceability': 0.8,
                    'acousticness': 0.2,
                    'instrumentalness': 0.1,
                    'user_age': 25,
                    'user_country': 1
                }
                
                prob = self.predict_repeat_listen(
                    user_id, song_id, current_timestamp, additional_features
                )
                
                predictions.append((song_id, prob))
        
        # Sort by probability and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [song_id for song_id, _ in predictions[:top_n]]


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)
    
    # Generate 1000 listening events
    n_events = 1000
    n_users = 100
    n_songs = 200
    
    # Create timestamps spanning 6 months
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2023, 7, 1)
    date_range = (end_date - start_date).total_seconds()
    
    # Generate random data with only numeric fields for date-related values to avoid aggregation issues
    data = {
        'user_id': np.random.randint(1, n_users + 1, n_events),
        'song_id': np.random.randint(1, n_songs + 1, n_events),
        'timestamp': [start_date.timestamp() + np.random.random() * date_range 
                      for _ in range(n_events)],
        'tempo': np.random.uniform(60, 180, n_events),
        'energy': np.random.random(n_events),
        'danceability': np.random.random(n_events),
        'acousticness': np.random.random(n_events),
        'instrumentalness': np.random.random(n_events),
        'user_age': np.random.randint(18, 65, n_events),
        'user_country': np.random.randint(1, 20, n_events)  # Encoded country
    }
    
    # Create DataFrame
    user_song_history = pd.DataFrame(data)
    
    # Initialize and train model
    recommender = SpotifyRecommendationSystem()
    metrics = recommender.train(user_song_history)
    
    print("Model performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Get recommendations for a user
    user_id = 1
    candidate_songs = list(range(1, 21))  # Consider songs 1-20
    recommendations = recommender.recommend_songs(user_id, candidate_songs, top_n=5)
    
    print(f"\nTop 5 recommendations for user {user_id}:")
    for i, song_id in enumerate(recommendations):
        print(f"{i+1}. Song {song_id}")


# In[5]:


pip install --upgrade pandas pyarrow scipy scikit-learn


# In[ ]:




