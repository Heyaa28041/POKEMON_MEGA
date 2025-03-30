import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

df = pd.read_csv("Pokemon1.csv", encoding="ISO-8859-1")
df["Mega_Evolution"] = df["Name"].apply(lambda x: "Yes" if "Mega" in x else "No")
features = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
X = df[features]
y = df["Mega_Evolution"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
y_test_bin = y_test.map({"No": 0, "Yes": 1})
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
precision, recall, _ = precision_recall_curve(y_test_bin, y_prob)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.show()
df["Predicted_Mega"] = model.predict(X)
df[["Name", "Predicted_Mega"]].rename(columns={"Name": "Pokemon", "Predicted_Mega": "Mega_Evolution"}).to_csv("Pokemon_Predictions_new.csv", index=False)
print("Predictions saved in 'Pokemon_Predictions_new.csv'")
