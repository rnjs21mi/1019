"""전체 실험 실행:  python main.py"""
import os, json, warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import load_trials, build_features
from model import get_models, train_all
from evaluation import holdout_table, confusion_analysis, cross_validate, paired_tests, cost_table
warnings.filterwarnings('ignore')
OUT = 'results'; os.makedirs(OUT, exist_ok=True)

trials, labels = load_trials()
FEAT = build_features(trials); nw = FEAT.shape[1]
print('시행 수', len(trials), '| 특징 배열', FEAT.shape, flush=True)

# 1) 시행 단위로 먼저 분할 (같은 시행의 겹치는 창이 양쪽에 들어가지 않게) -> 그 뒤 표준화
idx = np.arange(len(trials))
tr_i, te_i = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
assert not (set(tr_i) & set(te_i))
Xtr, ytr = FEAT[tr_i].reshape(-1, 25), np.repeat(labels[tr_i], nw)
Xte, yte = FEAT[te_i].reshape(-1, 25), np.repeat(labels[te_i], nw)
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)   # 학습셋에서만 fit

# 2) 학습 + hold-out 평가 + 혼동행렬
models = get_models(); train_all(models, Xtr, ytr)
df, preds = holdout_table(models, Xte, yte)
print(df.round(4)); df.round(4).to_csv(f'{OUT}/holdout.csv', index=False)
conf = confusion_analysis(preds, yte, OUT)

# 3) 5-fold CV + 통계 검정 + 계산 비용
fold_acc = cross_validate(models, FEAT, labels)
for k, v in fold_acc.items(): print(f'{k}: {np.mean(v)*100:.2f} ± {np.std(v)*100:.2f} %')
tests = paired_tests(fold_acc); [print(t) for t in tests]
cost = cost_table(models, Xtr, ytr, Xte); print(cost)

json.dump(dict(holdout=df.round(4).to_dict('records'), confusion=conf, fold_acc=fold_acc,
               tests=tests, cost=cost, n_train_windows=len(ytr), n_test_windows=len(yte)),
          open(f'{OUT}/results.json', 'w'), ensure_ascii=False, indent=1)
print('ALL DONE')
