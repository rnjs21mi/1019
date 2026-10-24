"""추가 점검: ZC 임계값(th)을 바꿔도 RF hold-out 정확도가 달라지는가?  python extra_zc_check.py"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data import load_trials, preprocess, make_windows, td_features, START, SEG_LEN
trials, labels = load_trials()
SEG = [make_windows(preprocess(t['x'])[START:START+SEG_LEN]) for t in trials]
tr_i, te_i = train_test_split(np.arange(len(trials)), test_size=0.2, stratify=labels, random_state=42)
for th in (0.0, 0.001, 0.01):
    F = np.stack([np.stack([td_features(w, th) for w in s]) for s in SEG])
    Xa, Xb = F[tr_i].reshape(-1, 25), F[te_i].reshape(-1, 25)
    ya, yb = np.repeat(labels[tr_i], 19), np.repeat(labels[te_i], 19)
    sc = StandardScaler(); Xa = sc.fit_transform(Xa); Xb = sc.transform(Xb)
    m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xa, ya)
    print(f'ZC th={th}: RF acc={accuracy_score(yb, m.predict(Xb))*100:.2f}%  (ZC==0 비율 {(F[:,:,15:20]==0).mean()*100:.1f}%)', flush=True)
