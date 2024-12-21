import pickle

with open("ssmc_fcm_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Mô hình đã được lưu vào 'ssmc_fcm_model.pkl'")
