# 음성 분류 경진대회

2022년 7월 11일 ~ 2022년 8월 5일동안 진행한 대회입니다. 

public 10등, private 14등으로 순위를 마무리했습니다.

댓글에 대한 평점을 분류하는 NLP 모델을 제작하는 대회였기 때문에, 우선 댓글에 대한 기본적인 이상한 값들 전처리를 진행한 이후, 댓글과 구어체 관련해서 학습을 진행해 더욱 좋은 성능을 보이는 KcElectra 모델을 이용해 학습했습니다.

다른 분들처럼 한국어->다른언어->한국어 처럼 번역을 통한 증강을 시도해 보지 못한 점이 아쉬웠던것 같습니다.


## 파일 구성

- Final Submission.ipynb [KcElectra using HuggingFace](./Final%20Submission.ipynb)

## Dacon에 작성한 글

KcElectra & Focal loss\
https://dacon.io/competitions/official/235938/codeshare/5941

## 데이터 출처
https://dacon.io/competitions/official/235938/data