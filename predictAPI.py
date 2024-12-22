import json
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS  # Import flask-cors module


class sSMC_FCM:
    def __init__(
        self, items, number_clusters, M, M1, alpha, Epsilon, max_iter, labeled_data
    ):
        self.items = items  # Dữ liệu đầu vào
        self.number_clusters = number_clusters  # Số cụm
        self.M = M  # Hệ số fuzziness cho dữ liệu không nhãn
        self.M1 = M1  # Hệ số fuzziness cho dữ liệu có nhãn
        self.alpha = alpha  # Hệ số điều chỉnh semi-supervised
        self.Epsilon = Epsilon  # Ngưỡng hội tụ
        self.max_iter = max_iter  # Số vòng lặp tối đa
        self.labeled_data = labeled_data  # {index: cluster_label}
        self.C = self.init_C_sSMC(
            self.items, self.number_clusters
        )  # Khởi tạo trung tâm cụm
        self.U = self.init_membership_matrix()  # Khởi tạo ma trận thành viên

    def init_C_sSMC(self, items, number_clusters):
        """Khởi tạo trung tâm cụm ban đầu"""
        C = np.zeros((number_clusters, len(items[0])))
        for i in range(number_clusters):
            C[i] = np.mean(items, axis=0)
        return C

    def init_membership_matrix(self):
        """Khởi tạo ma trận thành viên (membership matrix)"""
        U = np.random.dirichlet(np.ones(self.number_clusters), size=len(self.items))
        for idx, label in self.labeled_data.items():
            U[idx] = 0
            U[idx][label] = 1  # Khóa nhãn của dữ liệu đã gán
        return U

    def update_membership_matrix(self):
        """Cập nhật ma trận thành viên U"""
        for i in range(len(self.items)):
            if i in self.labeled_data:  # Bỏ qua điểm đã gán nhãn
                continue
            for j in range(self.number_clusters):
                denom = sum(
                    (
                        np.linalg.norm(self.items[i] - self.C[j])
                        / np.linalg.norm(self.items[i] - self.C[k])
                    )
                    ** (2 / (self.M - 1))
                    for k in range(self.number_clusters)
                )
                self.U[i][j] = 1 / denom

    def update_cluster_centers(self):
        """Cập nhật trung tâm cụm với ưu tiên các điểm có nhãn"""
        for j in range(self.number_clusters):
            # Tổng hợp từ điểm không nhãn
            unlabeled_numerator = np.sum(
                (self.U[:, j] ** self.M).reshape(-1, 1) * self.items, axis=0
            )
            unlabeled_denominator = np.sum(self.U[:, j] ** self.M)

            # Tổng hợp từ điểm có nhãn
            labeled_numerator = np.zeros_like(self.items[0])
            labeled_denominator = 0
            for idx, label in self.labeled_data.items():
                if label == j:  # Điểm thuộc cụm j
                    labeled_numerator += (self.U[idx, j] ** self.M1) * self.items[idx]
                    labeled_denominator += self.U[idx, j] ** self.M1

            # Cập nhật trung tâm cụm
            numerator = unlabeled_numerator + self.alpha * labeled_numerator
            denominator = unlabeled_denominator + self.alpha * labeled_denominator
            self.C[j] = numerator / denominator

    def run(self):
        """Chạy thuật toán"""
        for iteration in range(self.max_iter):
            previous_C = self.C.copy()
            self.update_membership_matrix()
            self.update_cluster_centers()
            # Kiểm tra hội tụ
            if np.linalg.norm(self.C - previous_C) < self.Epsilon:
                break

    def get_results(self):
        """Trả về kết quả"""
        labels = np.argmax(self.U, axis=1)
        # Gắn lại nhãn đúng cho các điểm đã có nhãn
        for idx, label in self.labeled_data.items():
            labels[idx] = label
        return {"cluster_centers": self.C, "labels": labels}


# Tải model từ file
model = pickle.load(open("ssmc_fcm_model.pkl", "rb"))

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)

labeled_data = {}


def preprocess_input_data(data):
    xdata = data.copy()
    scaler = MinMaxScaler()
    le = LabelEncoder()
    # Log transform 'Recency'
    # xdata['Recency'] = np.log(data['Recency'] + 2)  # Add smoothing constant
    xdata[["Frequency", "Monetary"]] = (
        scaler.fit_transform(data[["total_invoices", "total_price"]]) + 0.2
    ) * 10
    xdata["most_frequent_brand_numeric"] = (
        xdata["most_frequent_brand_numeric"] / 5 + 0.1
    )
    xdata["most_frequent_category_numeric"] = (
        xdata["most_frequent_category_numeric"] / 5 + 0.1
    )
    processed_data = xdata[
        [
            "Frequency",
            "Monetary",
            "most_frequent_brand_numeric",
            "most_frequent_category_numeric",
        ]
    ].values

    return processed_data


def preProcessData(df):

    # Đọc file JSON
    # with open(file_path, "r", encoding="utf-8") as file:
    #     data = json.load(file)  # data sẽ là một danh sách

    # Chuyển đổi danh sách thành DataFrame
    # df = pd.DataFrame(data)

    # In ra 5 dòng đầu tiên của DataFrame
    # print(df.head())
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
    summary.to_csv("data_1.csv", index=False)
    # Hiển thị kết quả
    summary["most_frequent_brand"] = summary["most_frequent_brand"].str.strip()
    # print(summary)
    return summary

    # Hiển thị kết quả
    # print(summary)
    summary.to_csv("data.csv", index=False)


@app.route("/getOderAnalysis", methods=["POST"])
def getOderAnalysis():
    try:
        labeled_data.clear()
        # Lấy dữ liệu đầu vào từ request (JSON)
        data = request.get_json()
        brand_df = pd.read_csv("brand.csv")  # Brand, Level, Numeric
        # Kiểm tra định dạng đầu vào
        if not isinstance(data, list):
            return jsonify({"error": "Input data must be a list of samples"}), 400

        # Chuyển dữ liệu thành numpy array
        data = pd.DataFrame(data)
        data = preProcessData(data)
        data = data.merge(
            brand_df[["Brand", "Level"]],
            left_on="most_frequent_brand",
            right_on="Brand",
            how="left",
        ).drop(columns=["most_frequent_brand", "Brand"])
        data.rename(columns={"Level": "most_frequent_brand"}, inplace=True)
        print(data)
        json_data = data.to_dict(orient="records")

        # Trả về kết quả dưới dạng JSON
        return jsonify(json_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu đầu vào từ request (JSON)
        data = request.get_json()

        # Kiểm tra định dạng đầu vào
        if not isinstance(data, list):
            return jsonify({"error": "Input data must be a list of samples"}), 400

        # Chuyển dữ liệu thành numpy array
        data = pd.DataFrame(data)
        summary = preProcessData(data)

        brand_df = pd.read_csv("brand.csv")  # Brand, Level, Numeric
        category_df = pd.read_csv("category.csv")  # Category, Numeric

        # Kiểm tra tên cột thực tế trong brand_df và category_df
        # print("Columns in brand_df:", brand_df.columns)
        # print("Columns in category_df:", category_df.columns)

        # Nếu tên cột có khoảng trắng hoặc vấn đề gì, bạn có thể sửa nó
        brand_df.columns = brand_df.columns.str.strip()  # Loại bỏ khoảng trắng thừa
        category_df.columns = (
            category_df.columns.str.strip()
        )  # Loại bỏ khoảng trắng thừa

        # Thay đổi 'most_frequent_brand' thành 'Numeric' từ brand_df
        summary = summary.merge(
            brand_df[["Brand", "Numeric"]],
            left_on="most_frequent_brand",
            right_on="Brand",
            how="left",
        ).drop(columns=["most_frequent_brand", "Brand"])
        summary.rename(columns={"Numeric": "most_frequent_brand_numeric"}, inplace=True)

        # Thay đổi 'most_frequent_category' thành 'Numeric' từ category_df
        summary = summary.merge(
            category_df[["Category", "Numeric"]],
            left_on="most_frequent_category",
            right_on="Category",
            how="left",
        ).drop(columns=["most_frequent_category", "Category"])
        summary.rename(
            columns={"Numeric": "most_frequent_category_numeric"}, inplace=True
        )

        # print(summary)

        X = preprocess_input_data(summary)
        print(summary)
        # Gọi hàm `fit` hoặc dự đoán dựa vào model
        model.items = X

        model.labeled_data = labeled_data

        model.U = model.init_membership_matrix()

        model.update_membership_matrix()

        model.run()

        predictions = model.get_results()  # Lấy nhãn dự đoán
        labels_predict = np.array(
            predictions["labels"]
        )  # Chuyển predictions thành numpy array
        print(labels_predict)
        labels_predict = labels_predict.astype(int)  # Đảm bảo predictions là int
        # Thay đổi giá trị dự đoán từ số thành label
        label_map = {0: "New", 1: "Normal", 2: "VIP"}
        predictions = np.vectorize(lambda x: label_map[x])(labels_predict)

        # Chuyển mảng numpy thành danh sách với CustomerID và Prediction
        result = [
            {
                "customer_id": summary["customer_id"].iloc[i],
                "Prediction": predictions[i],
            }
            for i in range(X.shape[0])
        ]

        # Trả về kết quả dưới dạng JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


label_mapping = {"New": 0, "Normal": 1, "VIP": 2}


@app.route("/updateLabel", methods=["POST"])
def update_label():
    try:
        # Lấy dữ liệu từ request JSON
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input, no data provided"}), 400

        # Lấy thông tin cần thiết
        index = data.get("index")
        label = data.get("label")
        print(index)
        # Kiểm tra dữ liệu hợp lệ
        if index is None or label is None:
            return jsonify({"error": "Missing 'index' or 'label' in the request"}), 400

        # Ánh xạ nhãn chuỗi sang số
        if label not in label_mapping:
            return (
                jsonify(
                    {
                        "error": f"Invalid label '{label}'. Must be one of {list(label_mapping.keys())}"
                    }
                ),
                400,
            )

        mapped_label = label_mapping[label]

        # Cập nhật dữ liệu vào biến labeled_data
        labeled_data[index] = mapped_label
        print(labeled_data)
        # Phản hồi thành công
        return (
            jsonify({"message": "Label updated successfully!", "data": labeled_data}),
            200,
        )

    except Exception as e:
        # Xử lý lỗi
        return jsonify({"error": str(e)}), 500


# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True, port=5000)
