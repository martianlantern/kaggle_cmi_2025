import lightgbm as lgb
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("../kaggle_cmi_2025/data/train.csv")
seq_labels = (
    train_data.groupby("sequence_id", sort=False)
    .agg(gesture=("gesture", "first"))
    .reset_index()
)
le = LabelEncoder()
le.fit(seq_labels["gesture"].astype(str))


features = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]

test = pd.read_csv("../kaggle_cmi_2025/data/test.csv")
test = test.fillna(0.0)

model = lgb.Booster(model_file="./outputs_lgbm/fold1.txt")

probs = model.predict(test[features])
pred_labels = np.argmax(probs, axis=1)

pred_gestures = le.inverse_transform(pred_labels)

submission = pd.DataFrame({
    "sequence_id": test["sequence_id"],
    "gesture": pred_gestures
})
submission.to_csv("./submission.csv", index=False)
print("Saved submission to ./submission.csv")
