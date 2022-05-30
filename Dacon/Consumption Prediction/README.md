# 소비자 데이터 기반 소비 예측 경진대회

2022년 5월 2일 ~ 2022년 5월 13일동안 진행한 대회입니다. 

public 2등, private 4등으로 약간의 순위가 내려가 4등으로 순위를 마무리했습니다.

전복 나이 예측처럼 여기서도 0~250 사이의 target이 많아, high target에 대한 예측을 더욱 학습했었다면 private에서 순위가 떨어지는것을 방지할 수 있지 않았을까 하는 생각도 듭니다.

Tree 모델로 진행하다가 성능이 더이상 오르지 않아, DNN 모델을 시도해보았는데, 성능이 더욱 올라갔습니다.

## 파일 구성

- Eda_with_Sub [dacon에 작성한 글에서 사용한 코드](./Eda_with_Sub.ipynb)
- Final Submission.ipynb [NN을 사용한 최종 제출 모델 코드](./Final%20Submission.ipynb)

## Dacon에 작성한 글

CAT + LGBM + XGB\
https://dacon.io/competitions/official/235893/codeshare/4880

DNN\
https://dacon.io/competitions/official/235893/codeshare/4962

## 데이터 출처
https://dacon.io/competitions/official/235893/data
