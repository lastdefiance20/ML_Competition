# 2022 암 예후예측 데이터 구축 AI 경진대회
## 암 융합 데이터를 이용한 암 예후 예측

### 코드 구조

```
$/
├── DATA/
│   ├── sample_data.csv # 서버 세팅 기간 동안 테스트를 위한 임의 파일
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── USER/baseline
    ├── config
    │   ├── config_predict.yaml
    │   ├── config_train.yaml
    ├── modules
    │   ├── trainer.py
    │   └── utils.py
    ├── readme.md
    ├── train.py
    └── predict.py
```

- config : 학습/추론에 필요한 파라미터 등을 기록하는 yml 파일
- utils
    - trainer.py : 에폭 별로 수행할 학습 과정
    - utils.py : log 기록, 여러 확장자 파일을 불러오거나 여러 확장자로 저장하는 등의 함수
- train.py : 학습 시 실행하는 코드
- predict.py : 추론 시 실행하는 코드 

### 필수 라이브러리
 - sklearn-pandas
 - pycox

### 사용 모델
- model : CoxPH

### 학습

1. `/USER/baseline/config/train.yaml` 수정
    1. DIRECTORY/dataset : 데이터 경로 지정(학습 데이터 위치한 디렉토리)
    2. 이외 파라미터 조정
2. `python train.py --train_serial [TRAIN_SERIAL]`
    1. train_serial 옵션 : 학습 결과 저장 경로(/USERS/results/train/{train_serial})
3. `/USER/baseline/results/train/` 내에 결과(weight, log, baseline hazards)가 저장됨


### 추론

1. `/USER/baseline/config/predict.yaml` 수정
    1. DIRECTORY/dataset : 데이터 경로 지정 (테스트 데이터 위치한 디렉토리)
    2. 이외 파라미터 조정
2. `python predict.py --train_serial train_serial`
    1. train_serial 옵션 : weight를 불러올 train serial (/USER/results/train 내 폴더명) 구도 별로 지정 - 
    2. train_serial : /USER/baseline/results/predict/ 내에 predict yaml 파일에 저장됨
3. `/USER/baseline/results/predict/` 내에 결과(weight,log, 제출파일)가 저장됨
