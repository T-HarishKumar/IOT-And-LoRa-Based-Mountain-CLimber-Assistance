import pandas as pd
import serial
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
file_path = "mountain_climbing_safety_data.csv"  
data = pd.read_csv('D:\SMVEC(2021-2025)\Final Year Project\Mountian\mountain_climbing_safety_data.csv')
X = data[['Heart_Rate', 'LM35', 'MEMS_X_out']]  
y = data['Output']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0)  # Silent training
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
ser = serial.Serial("COM5", 9600, timeout=1)  
time.sleep(2)  
print("\nWaiting for serial data...")
while True:
    try:
        if ser.in_waiting > 0:
            serial_data = ser.readline().decode('utf-8').strip()
            print(f"Received Data: {serial_data}")
            values = [v for v in serial_data.split(":") if v]

            if len(values) == 3:
                try:
                    heart_rate = float(values[0])
                    temperature = float(values[1])
                    accel_x = float(values[2])
                    input_data = np.array([[heart_rate, temperature, accel_x]])
                    input_data_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_data_scaled)[0]

                    if prediction == 1:
                        print("✅ Health Status: SAFE 🏔")
                    else:
                        print("⚠ Health Status: UNSAFE 🚨 - Take precautions!")

                except ValueError:
                    print("Error: Could not convert data to float. Check sensor values.")

            else:
                print("Invalid data format. Expected format: heartrate:temperature:AccelerometerX")

    except Exception as e:
        print(f"Error: {e}")

