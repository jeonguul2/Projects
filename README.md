# 🚀 Projects Archive

본 저장소는 개인적으로 진행하는 다양한 머신러닝 및 데이터 분석 프로젝트를 기록하는 공간입니다.

---

## 1. ⚽ 축구 경기 결과 예측 AI (fb-predict)
XGBoost 모델을 활용하여 축구 경기의 승/무/패를 예측하는 고성능 AI 프로그램입니다.

### 📌 주요 특징
* **데이터셋**: 약 17,000건의 실제 축구 경기 데이터 활용
* **핵심 지표**: 배당률, 환급률(Expected Value), 구매율 데이터를 통한 피처 엔지니어링
* **목표 성능**: 예측 정확도(Accuracy) 70% 이상 달성
* **버전 관리**:
  - `v4`: 초기 모델 구성 및 배당률 기반 확률 계산 로직 구현
  - `v5`: 환급률 자동 추출 및 데이터 전처리 고도화 (진행 중)
  - `v6`: 외부 데이터(팀 순위, 최근 전적) 결합 예정

### 🛠 Tech Stack
* Python, Pandas, XGBoost, Scikit-learn, BeautifulSoup(Crawling)

---

## 2. 🔊 Noise Machine Learning (G-XGBoost)
노이즈 데이터 분석 및 예측을 위한 G-XGBoost 기반 머신러닝 프로젝트입니다.

### 📌 주요 특징
* Noise Map 데이터를 활용한 패턴 분석 및 분류
* G-XGBoost 알고리즘 최적화를 통한 예측 모델 구현
* 데이터 전처리 및 노이즈 제거 알고리즘 적용

### 🛠 Tech Stack
* Python, XGBoost, NumPy, Matplotlib

---

## 📅 업데이트 기록
* **2024-XX-XX**: `Projects` 레포지토리 생성 및 `Noise ML` 초기 코드 업로드
* **2024-XX-XX**: `fb-predict` 프로젝트 시작 및 v4 구조 설계
