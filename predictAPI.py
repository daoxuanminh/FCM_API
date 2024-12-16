from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class FCM:
    def __init__(
        self,
        min_improvement,
        n_clusters,
        max_iterations,
        fuzzification_degree,
        dirichlet_concentration_params=None,
        metric="euclidean",
        verbose=False,
    ):
        self.objective = np.inf
        self.objective_history = []
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.n_clusters = n_clusters
        self.fuzzification_degree = fuzzification_degree
        self.dirichlet_concentration_params = dirichlet_concentration_params
        self.metric = self.__get_metric(metric)
        self.verbose = verbose
        self._vpc = np.nan

    def get_random_dirichlet_membership(self, dirichlet_concentration_params):

        if dirichlet_concentration_params is None:
            dirichlet_concentration_params = np.array([2.0] * self.n_clusters)

        assert (
            dirichlet_concentration_params.shape[0] == self.n_clusters
        ), f"dirichlet_concentration_params {dirichlet_concentration_params} does not match the number of clusters {self.n_clusters}"

        return np.random.dirichlet(dirichlet_concentration_params, size=self.n_samples)

    def set_membership_degree(self, membership_degree):
        self.membership_degree = membership_degree

    def fit(self, X, init_membership_degree=None):
        assert isinstance(X, np.ndarray), "input of fit method should be a numpy array"
        self.X = X
        self.n_samples, self.n_features = self.X.shape
        self.distance_matrix = np.zeros(shape=(self.n_samples, self.n_clusters))
        if init_membership_degree is None:
            init_membership_degree = self.get_random_dirichlet_membership(
                self.dirichlet_concentration_params
            )
        self.set_membership_degree(init_membership_degree)
        self.normalize_membership_degree()

        iteration = 0
        while True:
            self.update_centers()
            self.update_distances()
            self.update_membership_degree()
            self.update_objective()
            if self.stopping_criterion(iteration):
                print(self.MCD(), self.xi_beni(), self.lables())
                break
            iteration += 1
            if self.verbose:
                print(self.vpc())

    def update_centers(self):
        self.fuzzy_membership_degree = self.membership_degree**self.fuzzification_degree
        self.centers = (self.fuzzy_membership_degree.T @ self.X) / np.expand_dims(
            self.fuzzy_membership_degree.sum(axis=0), 1
        )

    def update_objective(self):
        self.objective = (
            self.fuzzy_membership_degree * self.distance_matrix_squared
        ).sum()
        self.objective_history.append(self.objective)

    def stopping_criterion(self, iteration):
        if iteration > self.max_iterations:
            return True

        if iteration > 2:
            self.improvement = np.abs(
                self.objective_history[-2] - self.objective_history[-1]
            )
            return self.improvement < self.min_improvement

        return False

    def update_distances(self):
        for center_idx in range(self.n_clusters):
            self.distance_matrix[:, center_idx] = self.metric(
                np.expand_dims(self.centers[center_idx], 0), self.X
            )

    def update_membership_degree(self):
        self.distance_matrix_squared = self.distance_matrix**2
        for center_idx in range(self.n_clusters):
            for sample_idx in range(self.n_samples):
                temp = (
                    self.distance_matrix_squared[sample_idx, center_idx]
                    / self.distance_matrix_squared[sample_idx, :]
                )
                self.membership_degree[sample_idx, center_idx] = 1 / (
                    (temp.sum()) ** (2 / (self.fuzzification_degree - 1))
                )
        self.normalize_membership_degree()

    def normalize_membership_degree(self):
        self.membership_degree = self.membership_degree / self.membership_degree.sum(
            axis=1, keepdims=True
        )

    def __get_metric(self, name):
        metrics = {"euclidean": self._euclid_metric, "cosine": self.__cosine_metric}
        return metrics[name]

    def _euclid_metric(self, x, y):
        return np.linalg.norm(x - y, axis=1)

    def __cosine_metric(self, x, y):
        return 1 - np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))

    def xi_beni(self):
        b = []
        for i in self.centers:
            for j in self.centers:
                b.append((np.linalg.norm(i - j)) ** 2)
            if 0 in b:
                b.remove(0)
        b.sort()
        c = b[0]
        self.a = (self.n_samples) * c
        return self.a

    def MCD(self):
        M = []
        for i in self.centers:
            for j in self.centers:
                M.append((np.linalg.norm(i - j)) ** 2)
                if 0 in M:
                    M.remove(0)
        M.sort()
        self.F = M[0]
        return self.F

    def update_centers2(self):
        self.fuzzy_membership_degree = self.membership_degree**self.fuzzification_degree
        self.centers = (self.fuzzy_membership_degree.T @ self.X) / np.expand_dims(
            self.fuzzy_membership_degree.sum(axis=0), 1
        )
        return self.centers

    def lables(self):
        self.predict = np.argmax(self.membership_degree, axis=1)
        return self.predict


# Tải model từ file
model = pickle.load(open("model.pkl", "rb"))

# Khởi tạo ứng dụng Flask
app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"message": "API for FCM Model is running!"})


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
        data = process_data(data)
        print(data["CustomerID"])
        X = preprocess_input_data(data)
        # print(X[:5])
        # Gọi hàm `fit` hoặc dự đoán dựa vào model
        model.fit(X)
        predictions = model.lables().tolist()  # Lấy nhãn dự đoán
        predictions = np.array(predictions)  # Chuyển predictions thành numpy array
        predictions = predictions.astype(int)  # Đảm bảo predictions là int

        # Thay đổi giá trị dự đoán từ số thành label
        label_map = {0: "VIP", 1: "Normal", 2: "New"}
        predictions = np.vectorize(lambda x: label_map[x])(predictions)

        # Chuyển mảng numpy thành danh sách với CustomerID và Prediction
        result = [
            {"customer_id": data["CustomerID"].iloc[i], "Prediction": predictions[i]}
            for i in range(X.shape[0])
        ]

        # Trả về kết quả dưới dạng JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Read JSON file into a DataFrame
df = pd.read_json("data.json")

print(df.head)

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True, port=5000)
