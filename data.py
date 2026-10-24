"""데이터 로딩 + 전처리 + 특징 추출 (시행 단위로 먼저 다룬다)"""
import os, glob, subprocess
import numpy as np, pandas as pd
from scipy.signal import butter, iirnotch, filtfilt

REPO = 'https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset.git'
REPO_DIR = 'sEMG-Hand-Gesture-Elbow-Angle-Dataset'
FS, TRIAL_LEN = 1000, 5000
START, SEG_LEN, WIN, HOP = 1000, 2000, 200, 100      # 분석구간 1.0~3.0s, 200ms 창, 50% 오버랩
CLASS_NAMES = [f'{g}_{a}' for a in (30, 60) for g in ['fist', 'open', 'pro', 'extor']]

def ensure_data():
    if not os.path.exists(REPO_DIR):
        subprocess.run(['git', 'clone', '--depth', '1', REPO, REPO_DIR], check=True)
    return os.path.join(REPO_DIR, 'data')

def load_trials():
    """CSV 1개 = 30회 반복 연속 기록. mark 열 기준으로 5초(5000샘플) 시행으로 자른다 (총 1920 시행)."""
    out = []
    for f in sorted(glob.glob(os.path.join(ensure_data(), '*.csv'))):
        b = os.path.basename(f)[:-4]; subj, ang, g = b[0], int(b[2:4]), b[5:]
        d = pd.read_csv(f); x = d.iloc[:, :5].values.astype(float)
        for t, s in enumerate(np.where(d['mark'].values != 0)[0]):
            out.append(dict(x=x[s:s+TRIAL_LEN], subj=subj, angle=ang, gesture=g, trial=t))
    labels = np.array([CLASS_NAMES.index(f"{t['gesture']}_{t['angle']}") for t in out])
    return out, labels

def preprocess(x, fs=FS):
    """60Hz 노치(Q=30) + 20~450Hz 4차 Butterworth 대역통과, filtfilt(영위상)"""
    bn, an = iirnotch(60, 30, fs); x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20/(fs/2), 450/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)

def make_windows(x, win=WIN, hop=HOP):
    n = (len(x) - win) // hop + 1
    return np.stack([x[i*hop:i*hop+win] for i in range(n)])

def td_features(w, th=0.01):
    """w:(200,5) -> 25차원 = RMS, MAV, WL, ZC, SSC x 5채널"""
    rms = np.sqrt(np.mean(w**2, axis=0)); mav = np.mean(np.abs(w), axis=0)
    d = np.diff(w, axis=0); wl = np.sum(np.abs(d), axis=0)
    zc = np.sum(((w[:-1]*w[1:]) < 0) & (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1]*d[1:]) < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])

def build_features(trials):
    """시행별 특징 (N_trials, 19, 25), 표준화 전. 표준화는 분할 뒤 학습셋에서만 fit."""
    return np.stack([np.stack([td_features(w) for w in
                    make_windows(preprocess(t['x'])[START:START+SEG_LEN])]) for t in trials])
