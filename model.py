"""모델 구성 (같은 random_state=42). SVM·RF는 기본값 위주, XGBoost만 지정 설정 -> 비대칭 비교"""
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models():
    return {
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr'),
        'RF':  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGB': XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.8, random_state=42),
    }

def train_all(models, Xtr, ytr):
    times = {}
    for k, m in models.items():
        t0 = time.time(); m.fit(Xtr, ytr); times[k] = time.time() - t0
        print(f'{k} 학습 완료 ({times[k]:.1f}s)', flush=True)
    return times
