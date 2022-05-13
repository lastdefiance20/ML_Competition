# 전복 나이 예측 경진대회

2022년 3월 21일 ~ 2022년 4월 1일동안 진행한 대회입니다. 

public 52등, private 12등으로 엄청난 순위향상이 이루어져 최종 12등으로 대회를 마무리했습니다.

모델이 high target, 즉 나이가 많은 전복에 대한 예측을 잘 못하는것 같아 샘플이 부족한 부분을 oversampling 해주었습니다.\
이렇게 모델이 편향된 학습을 하지 않도록 샘플링 해준 부분이 급격한 순위상승으로 이루어졌다고 생각합니다.

아마 public에서 사용한 데이터셋에서는 나이가 많은 전복에 대한 샘플이 별로 없었던것 같았던것 같습니다.\ 
따라서 public 점수만 믿고 튜닝한 분들은 모델이 편향된 학습을 해서 private에서는 대거 점수가 내려간게 아닐까 생각합니다.

Data Oversampling 기법을 배울 수 있는 대회였습니다.

## 파일 구성

- Data Analysis with Insight.ipynb [Tree 모델과 MLP 모델을 이용한 Data Analysis 초기 코드](./Data%20Analysis%20with%20Insight.ipynb)

- Final Submission.ipynb [NN을 이용한 최종 제출 모델 코드](./Final%20Submission.ipynb)

## Dacon에 작성한 글

MLP + NGB + XGB + CAT with Data Over Sampling\
https://dacon.io/competitions/official/235877/codeshare/4711

## 데이터 출처
https://dacon.io/competitions/official/235877/data
