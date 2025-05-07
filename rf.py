from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score

X,y = None

X_train, X_test, y_train, y_test = None #train_test_split(X, y, stratify=y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f2 = fbeta_score(y_test, y_pred, beta=2)
print(f"F2 Score: {f2:.4f}")