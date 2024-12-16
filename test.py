import pandas as pd

# Giả sử đây là DataFrame ban đầu (X) với các CustomerID
X = pd.DataFrame(
    {
        "CustomerID": ["C001", "C002", "C003", "C004", "C005"],
        "Feature1": [1.2, 2.3, 3.1, 4.5, 5.1],
        "Feature2": [0.5, 0.7, 1.1, 1.5, 1.9],
    }
)

# Dự đoán từ mô hình (dạng list)
predictions = [0, 1, 1, 0, 2]  # Dự đoán ví dụ

# Mapping CustomerID với predictions
X["Prediction"] = predictions

# Thay thế giá trị dự đoán: 0 -> 'VIP', 1 -> 'Normal', 2 -> 'New'
X["Prediction"] = X["Prediction"].map({0: "VIP", 1: "Normal", 2: "New"})

# Chỉ lấy CustomerID và Prediction
result = X[["CustomerID", "Prediction"]].to_dict(orient="records")

# Kết quả
print(result)
