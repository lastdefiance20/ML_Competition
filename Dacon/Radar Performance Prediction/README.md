# LG AI 경진대회 - 해커톤 팀

## 팀명 :  해커톤

- 팀장 정재윤
- 팀원 Kade Na, 강보근

> Presentation : [LG_radar.pdf](./LG_radar.pdf)

## 코드 실행 환경
> ## run at paperspace gradient
> **Start from paperspace gradient Scratch**
> https://hub.docker.com/r/paperspace/gradient-base \
> \
> **docker tag** \
> pt112-tf29-jax0314-py39-20220803
> ## machine env
> 8 cpu, A4000 gpu

## 파일 목록
```
┖ model - 모델 파일이 저장된 폴더
    ┖ cat_1
    ┖ cat_2
    ┖ ...
┖ Final_code.ipynb - 전체 코드 실행 파일
┖ test.csv
┖ train.csv
┖ y_feature_spec_info.csv
┖ sample_submission.csv
```
데이터의 경우에는 파일 목록에 맞게 위치시켜주면 됩니다.

## 전체 실행 프로세스
**Final Code 내부 실행 프로세스**
1. 데이터 불러오기
1. 데이터 분석
1. 데이터 전처리
1. 모델 생성
1. 모델 학습
1. 결과 제출

## 학습 방법
1. Run All code in **Final_Code.ipynb**
2. Submit final_pred.csv
