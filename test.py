import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import LabelEncoder


def preprocess_input_data(data):
    xdata = data.copy()
    scaler = MinMaxScaler()
    le = LabelEncoder()
    # Log transform 'Recency'
    xdata["Recency"] = np.log(data["Recency"] + 2)  # Add smoothing constant
    xdata[["Frequency", "Monetary"]] = (
        scaler.fit_transform(data[["Frequency", "Monetary"]]) * 10
    )
    xdata["shopping_mall"] = le.fit_transform(data["shopping_mall"]) / 5
    processed_data = xdata[
        ["Recency", "Frequency", "Monetary", "shopping_mall", "gender"]
    ].values

    return processed_data


def process_data(df):
    # Recency
    df["date"] = pd.DatetimeIndex(df["invoice_date"]).date
    recency = df.groupby(by="customer_id", as_index=False)["date"].max()
    recency.columns = ["CustomerID", "LastPurchasedDate"]
    recency["LastPurchasedDate"] = pd.to_datetime(recency["LastPurchasedDate"])
    max_date = recency["LastPurchasedDate"].max()
    recency["Recency"] = (max_date - recency["LastPurchasedDate"]).dt.days

    # Frequency
    frequency_df = df.groupby(by=["customer_id"], as_index=False)["invoice_no"].count()
    frequency_df.columns = ["CustomerID", "Frequency"]

    # Monetary
    monetary_df = df.groupby("customer_id", as_index=False).agg({"price": "sum"})
    monetary_df.columns = ["CustomerID", "Monetary"]

    # Merge gender
    temp_df = recency.merge(frequency_df, on="CustomerID")
    customer_gender_map = df[["customer_id", "gender"]].drop_duplicates()
    gender_dict = (
        customer_gender_map.set_index("customer_id")["gender"]
        .map({"Men": 0, "Women": 1})
        .to_dict()
    )
    temp_df["gender"] = temp_df["CustomerID"].map(gender_dict)
    temp_df["gender"].fillna(0.5, inplace=True)

    rfm_df = temp_df.merge(monetary_df, on="CustomerID")
    rfm_df.set_index("CustomerID", inplace=True)

    mall_spending = df.groupby(["customer_id", "shopping_mall"])["price"].sum()

    # Tìm trung tâm mua sắm mà mỗi khách hàng chi tiêu nhiều nhất
    most_spent_mall = mall_spending.groupby("customer_id").idxmax()

    # Chỉ lấy tên của shopping_mall từ tuple
    most_spent_mall = most_spent_mall.apply(lambda x: x[1])

    # Tạo DataFrame chứa kết quả
    result = most_spent_mall.reset_index()
    result.columns = ["CustomerID", "shopping_mall"]

    # Hiển thị kết quả
    result_mall = pd.DataFrame(result)
    temp_df = temp_df.merge(result_mall, how="left", on="CustomerID")
    rfm_df = temp_df.merge(monetary_df, on="CustomerID")

    return rfm_df


import pandas as pd
import json

# Đường dẫn đến file data.json
file_path = "data.json"

# Đọc file JSON
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)  # data sẽ là một danh sách

# Chuyển đổi danh sách thành DataFrame
df = pd.DataFrame(data)

# In ra 5 dòng đầu tiên của DataFrame
print(df.head())
summary = (
    df.groupby("customer_id")
    .agg(
        total_price=("price", "sum"),  # Tổng giá
        most_frequent_category=(
            "category",
            lambda x: x.value_counts().idxmax(),
        ),  # Danh mục mua nhiều nhất
        most_frequent_shopping_mall=(
            "shopping_mall",
            lambda x: x.value_counts().idxmax(),
        ),  # Trung tâm mua sắm nhiều nhất
        total_invoices=("invoice_no", "count"),  # Tổng số đơn hàng
    )
    .reset_index()
)
summary.rename(
    columns={"most_frequent_shopping_mall": "most_frequent_brand"}, inplace=True
)
# Hiển thị kết quả
summary["most_frequent_brand"] = summary["most_frequent_brand"].str.strip()
print(summary)
brand_df = pd.read_csv("brand.csv")  # Brand, Level, Numeric
category_df = pd.read_csv("category.csv")  # Category, Numeric

# Kiểm tra tên cột thực tế trong brand_df và category_df
print("Columns in brand_df:", brand_df.columns)
print("Columns in category_df:", category_df.columns)

# Nếu tên cột có khoảng trắng hoặc vấn đề gì, bạn có thể sửa nó
brand_df.columns = brand_df.columns.str.strip()  # Loại bỏ khoảng trắng thừa
category_df.columns = category_df.columns.str.strip()  # Loại bỏ khoảng trắng thừa

# Thay đổi 'most_frequent_brand' thành 'Numeric' từ brand_df
summary = summary.merge(
    brand_df[["Brand", "Numeric"]],
    left_on="most_frequent_brand",
    right_on="Brand",
    how="left",
).drop(columns=["most_frequent_brand", "Brand"])
summary.rename(columns={"Numeric": "most_frequent_brand_Numeric"}, inplace=True)

# Thay đổi 'most_frequent_category' thành 'Numeric' từ category_df
summary = summary.merge(
    category_df[["Category", "Numeric"]],
    left_on="most_frequent_category",
    right_on="Category",
    how="left",
).drop(columns=["most_frequent_category", "Category"])
summary.rename(columns={"Numeric": "most_frequent_category_Numeric"}, inplace=True)

# Hiển thị kết quả
print(summary)
summary.to_csv("data.csv", index=False)
