import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_disc_sight_model():
    # 1. Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('ultimate_biomechanics_data.csv')

    # 2. Separate features (X) and label (y)
    X = df.drop('label', axis=1)
    y = df['label']

    # 3. Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize the Random Forest model
    # n_estimators=100 provides a good balance between speed and performance
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 5. Train the model
    print("Training model...")
    model.fit(X_train, y_train)

    # 6. Evaluate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nTraining Complete!")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # 7. Save the trained model
    joblib.dump(model, 'ultimate_form_model.pkl')
    print("Model saved as 'ultimate_form_model.pkl'")

if __name__ == "__main__":
    train_disc_sight_model()