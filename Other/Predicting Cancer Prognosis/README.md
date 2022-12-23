# 2022 암 예후예측 데이터 구축 AI 경진대회
## Predicting Cancer Prognosis Using Cancer Fusion Data

### code structure

```
DATA/
    sample/
        sample_data.csv
    train/
        train.csv
    test/
        test.csv
    sample_submission.csv
USER/RESULT
    config
        config_train.yaml
    modules
        utils.py
    xgbse
        library for use xgbse
    readme.md
    ensemble_train.py
    requirements.txt
```

- config : yaml file that contain parameter for train & pred
- utils.py : record terminal log
- ensemble_train.py : code for train & predict

### model
- model : xgbse

### train with pred
1. run code using `python ensemble_train.py`

2. result will be save in /USER/RESLUT/2022~/result_df.csv