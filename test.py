from pymongo import MongoClient
from datetime import datetime, timedelta

# Kết nối đến MongoDB
client = MongoClient("mongodb://localhost:27017")  # Đảm bảo sửa URL nếu cần
db = client["shop"]  # Thay 'your_database_name' bằng tên database của bạn
collection = db["Orders"]  # Thay 'your_collection_name' bằng tên collection của bạn

# Lấy ngày hôm nay (0 giờ sáng)
today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

# Truy vấn để xóa các bản ghi có 'createdAt' nhỏ hơn ngày hôm nay
result = collection.delete_many({"createdAt": {"$lt": today}})

# In ra số lượng bản ghi đã xóa
print(f"Deleted {result.deleted_count} documents.")
