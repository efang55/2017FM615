
# Kobe Bryant shot selection 

### 載入套件及data


```python
import numpy as np 
import pandas as pd 
import seaborn as sns
import math as m
import matplotlib.pyplot as plt
%matplotlib inline
 
from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation


sns.set_style('whitegrid')
pd.set_option('display.max_columns', None) # display all columns
```


```python
df = pd.read_csv('data.csv', encoding='BIG5')
not_applicable = df['shot_made_flag'].isnull()

df = df[~not_applicable]
```


```python
pd.DataFrame(df.head())
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>action_type</th>
      <th>combined_shot_type</th>
      <th>game_event_id</th>
      <th>game_id</th>
      <th>lat</th>
      <th>loc_x</th>
      <th>loc_y</th>
      <th>lon</th>
      <th>minutes_remaining</th>
      <th>period</th>
      <th>playoffs</th>
      <th>season</th>
      <th>seconds_remaining</th>
      <th>shot_distance</th>
      <th>shot_made_flag</th>
      <th>shot_type</th>
      <th>shot_zone_area</th>
      <th>shot_zone_basic</th>
      <th>shot_zone_range</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>game_date</th>
      <th>matchup</th>
      <th>opponent</th>
      <th>shot_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>12</td>
      <td>20000012</td>
      <td>34.0443</td>
      <td>-157</td>
      <td>0</td>
      <td>-118.4268</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>2000-01</td>
      <td>22</td>
      <td>15</td>
      <td>0.0</td>
      <td>2PT Field Goal</td>
      <td>Left Side(L)</td>
      <td>Mid-Range</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000-10-31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>35</td>
      <td>20000012</td>
      <td>33.9093</td>
      <td>-101</td>
      <td>135</td>
      <td>-118.3708</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>2000-01</td>
      <td>45</td>
      <td>16</td>
      <td>1.0</td>
      <td>2PT Field Goal</td>
      <td>Left Side Center(LC)</td>
      <td>Mid-Range</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000-10-31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>43</td>
      <td>20000012</td>
      <td>33.8693</td>
      <td>138</td>
      <td>175</td>
      <td>-118.1318</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2000-01</td>
      <td>52</td>
      <td>22</td>
      <td>0.0</td>
      <td>2PT Field Goal</td>
      <td>Right Side Center(RC)</td>
      <td>Mid-Range</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000-10-31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Driving Dunk Shot</td>
      <td>Dunk</td>
      <td>155</td>
      <td>20000012</td>
      <td>34.0443</td>
      <td>0</td>
      <td>0</td>
      <td>-118.2698</td>
      <td>6</td>
      <td>2</td>
      <td>0</td>
      <td>2000-01</td>
      <td>19</td>
      <td>0</td>
      <td>1.0</td>
      <td>2PT Field Goal</td>
      <td>Center(C)</td>
      <td>Restricted Area</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000-10-31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>244</td>
      <td>20000012</td>
      <td>34.0553</td>
      <td>-145</td>
      <td>-11</td>
      <td>-118.4148</td>
      <td>9</td>
      <td>3</td>
      <td>0</td>
      <td>2000-01</td>
      <td>32</td>
      <td>14</td>
      <td>0.0</td>
      <td>2PT Field Goal</td>
      <td>Left Side(L)</td>
      <td>Mid-Range</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>Los Angeles Lakers</td>
      <td>2000-10-31</td>
      <td>LAL @ POR</td>
      <td>POR</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25697 entries, 1 to 30696
    Data columns (total 25 columns):
    action_type           25697 non-null object
    combined_shot_type    25697 non-null object
    game_event_id         25697 non-null int64
    game_id               25697 non-null int64
    lat                   25697 non-null float64
    loc_x                 25697 non-null int64
    loc_y                 25697 non-null int64
    lon                   25697 non-null float64
    minutes_remaining     25697 non-null int64
    period                25697 non-null int64
    playoffs              25697 non-null int64
    season                25697 non-null object
    seconds_remaining     25697 non-null int64
    shot_distance         25697 non-null int64
    shot_made_flag        25697 non-null float64
    shot_type             25697 non-null object
    shot_zone_area        25697 non-null object
    shot_zone_basic       25697 non-null object
    shot_zone_range       25697 non-null object
    team_id               25697 non-null int64
    team_name             25697 non-null object
    game_date             25697 non-null object
    matchup               25697 non-null object
    opponent              25697 non-null object
    shot_id               25697 non-null int64
    dtypes: float64(3), int64(11), object(11)
    memory usage: 5.1+ MB



```python
df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_event_id</th>
      <th>game_id</th>
      <th>lat</th>
      <th>loc_x</th>
      <th>loc_y</th>
      <th>lon</th>
      <th>minutes_remaining</th>
      <th>period</th>
      <th>playoffs</th>
      <th>seconds_remaining</th>
      <th>shot_distance</th>
      <th>shot_made_flag</th>
      <th>team_id</th>
      <th>shot_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>25697.000000</td>
      <td>2.569700e+04</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>25697.000000</td>
      <td>2.569700e+04</td>
      <td>25697.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>249.348679</td>
      <td>2.474109e+07</td>
      <td>33.953043</td>
      <td>7.148422</td>
      <td>91.257345</td>
      <td>-118.262652</td>
      <td>4.886796</td>
      <td>2.520800</td>
      <td>0.146243</td>
      <td>28.311554</td>
      <td>13.457096</td>
      <td>0.446161</td>
      <td>1.610613e+09</td>
      <td>15328.166946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>149.778520</td>
      <td>7.738108e+06</td>
      <td>0.088152</td>
      <td>110.073147</td>
      <td>88.152106</td>
      <td>0.110073</td>
      <td>3.452475</td>
      <td>1.151626</td>
      <td>0.353356</td>
      <td>17.523392</td>
      <td>9.388725</td>
      <td>0.497103</td>
      <td>0.000000e+00</td>
      <td>8860.462397</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>2.000001e+07</td>
      <td>33.253300</td>
      <td>-250.000000</td>
      <td>-44.000000</td>
      <td>-118.519800</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.610613e+09</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>111.000000</td>
      <td>2.050006e+07</td>
      <td>33.884300</td>
      <td>-67.000000</td>
      <td>4.000000</td>
      <td>-118.336800</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>1.610613e+09</td>
      <td>7646.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>253.000000</td>
      <td>2.090034e+07</td>
      <td>33.970300</td>
      <td>0.000000</td>
      <td>74.000000</td>
      <td>-118.269800</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>1.610613e+09</td>
      <td>15336.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>367.000000</td>
      <td>2.960027e+07</td>
      <td>34.040300</td>
      <td>94.000000</td>
      <td>160.000000</td>
      <td>-118.175800</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>43.000000</td>
      <td>21.000000</td>
      <td>1.000000</td>
      <td>1.610613e+09</td>
      <td>22976.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>653.000000</td>
      <td>4.990009e+07</td>
      <td>34.088300</td>
      <td>248.000000</td>
      <td>791.000000</td>
      <td>-118.021800</td>
      <td>11.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>59.000000</td>
      <td>79.000000</td>
      <td>1.000000</td>
      <td>1.610613e+09</td>
      <td>30697.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(include=['object', 'category'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>action_type</th>
      <th>combined_shot_type</th>
      <th>season</th>
      <th>shot_type</th>
      <th>shot_zone_area</th>
      <th>shot_zone_basic</th>
      <th>shot_zone_range</th>
      <th>team_name</th>
      <th>game_date</th>
      <th>matchup</th>
      <th>opponent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
      <td>25697</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>55</td>
      <td>6</td>
      <td>20</td>
      <td>2</td>
      <td>6</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
      <td>1558</td>
      <td>74</td>
      <td>33</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Jump Shot</td>
      <td>Jump Shot</td>
      <td>2005-06</td>
      <td>2PT Field Goal</td>
      <td>Center(C)</td>
      <td>Mid-Range</td>
      <td>Less Than 8 ft.</td>
      <td>Los Angeles Lakers</td>
      <td>2016-04-13</td>
      <td>LAL @ SAS</td>
      <td>SAS</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>15836</td>
      <td>19710</td>
      <td>1924</td>
      <td>20285</td>
      <td>11289</td>
      <td>10532</td>
      <td>7857</td>
      <td>25697</td>
      <td>43</td>
      <td>853</td>
      <td>1638</td>
    </tr>
  </tbody>
</table>
</div>



### Data視覺化


```python
from IPython.display import Image
from IPython.display import display
display(Image(url='https://i.imgur.com/IykGbhG.jpg'))
```


<img src="https://i.imgur.com/IykGbhG.jpg"/>


根據上圖可以看出我們布萊恩先生最愛的射籃型態是jump shot，但其命中率只有39.1%；而扣除掉dunk，布萊恩先生命中率最高的是bank shot，但只佔他出手比率的0.5，%顯然不喜歡這種投射姿勢><


```python
display(Image(url='https://i.imgur.com/r6PRwR3.jpg'))
```




![jpeg](Final_project_files/Final_project_11_0.jpeg)



根據此圖可看出老大較喜歡在球場左邊進行攻擊，或許這和他慣用手為右手有關。而且儘管老大較常在左邊做出手，他的命中率也不遜於右側，所以下次和Kobe在球場上較勁時，記得踩住他右切的路線，不然會被他射得不要不要的。<br/>
再來是老大統治NBA的時代，三分線瘋狂出手及過了半場旱地拔蔥Logo shot尚未盛行，因此他在三分線外的命中率也不算太高。


```python
display(Image(url='https://i.imgur.com/jdjEJcM.jpg', width="400"))
```




![jpeg](Final_project_files/Final_project_13_0.jpeg)



根據此圖，老大的投射命中率隨投射距離明顯地遞減，在16ft.-24ft.間他瘋狂射籃，射了將近7000發，但命中率僅40.2%，而印象派球迷也常詬病他中距離的出手選擇，以實際數據分析結果也是相同結果。


```python
display(Image(url='https://i.imgur.com/SDiLPM8.jpg'))
```




![jpeg](Final_project_files/Final_project_15_0.jpeg)



在2008-09賽季Kobe老大得到了NBA MVP的殊榮，當年他幹進了生涯第二高的1864分，2分球命中率49.6%、3分球命中率為35.5%，效率頗高，可說是魅力四射、風靡萬眾的一年<br/>
在2012-13賽季末段可謂是老大迷心碎的一年，心裡有如他阿基里斯腱般被狠狠撕裂，也自此賽季過後，他身手不再，從此走下神壇。

### Data及特徵整理


```python
df["game_year"] = df["game_date"].str[0:4].astype(int)
df["game_month"] = df["game_date"].str[5:7].astype(int)
```


```python
df['action_first_words'] = df["action_type"].str.split(' ').str[0]
```


```python
df['season_start_year'] = df.season.str.split('-').str[0].astype(int)
```


```python
df["remaining"] = df["minutes_remaining"] * 60 + df["seconds_remaining"]
```


```python
df['away'] = df.matchup.str.contains('@')
```


```python
df['distance_bin'] = pd.cut(df.shot_distance, bins=10, labels=range(10))
```


```python
df['angle'] = df.apply(lambda row: 90 if row['loc_y']==0 else m.degrees(m.atan(row['loc_x']/abs(row['loc_y']))),axis=1)
df['angle_bin'] = pd.cut(df.angle, 7, labels=range(7))
df['angle_bin'] = df.angle_bin.astype(int)
```

### 選取訓練、測試集，並作第一次Features選取


```python
train = df.copy()
test = pd.DataFrame(df[['shot_made_flag', 'shot_id']].copy())
```


```python
selected_features = ['action_first_words', 'combined_shot_type', 'remaining', 'period', 'season_start_year'
    , 'shot_type', 'shot_zone_basic', 'shot_zone_range', 'game_year',
    'game_month', 'opponent', 'away', 'distance_bin', 'angle_bin', 'shot_id']
```


```python
train = train[selected_features]
train = pd.get_dummies(train)
```


```python
# 設定訓練集、切割測試集
testid = [30085, 26369, 27841, 15946, 22900]

X_train = train[(train.shot_id != testid[0]) & (train.shot_id != testid[1]) & (train.shot_id != testid[2]) & (train.shot_id != testid[3]) & (train.shot_id != testid[4])]
Y_train = test[(test.shot_id != testid[0]) & (test.shot_id != testid[1]) & (test.shot_id != testid[2]) & (test.shot_id != testid[3]) & (test.shot_id != testid[4])].drop('shot_id', axis = 1)

X_test_1 = train[train.shot_id == testid[0]]
X_test_2 = train[train.shot_id == testid[1]]
X_test_3 = train[train.shot_id == testid[2]]
X_test_4 = train[train.shot_id == testid[3]]
X_test_5 = train[train.shot_id == testid[4]]
```

### 用random forest再進行第二次features選取


```python
rfc = RandomForestClassifier()
```


```python
rfc.fit(X_train, Y_train)
```

    C:\Users\fang\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      """Entry point for launching an IPython kernel.





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False)




```python
features_good_fit = pd.DataFrame(rfc.feature_importances_, index=X_train.columns, columns=["good_fit"])
top50features = features_good_fit.sort_values("good_fit", ascending=False).head(50).index
```


```python
top50features
```




    Index(['remaining', 'shot_id', 'angle_bin', 'game_year', 'season_start_year',
           'game_month', 'period', 'action_first_words_Jump', 'away',
           'action_first_words_Driving', 'action_first_words_Layup',
           'combined_shot_type_Jump Shot', 'combined_shot_type_Dunk',
           'action_first_words_Running', 'opponent_SAC', 'opponent_HOU',
           'opponent_DEN', 'opponent_GSW', 'opponent_POR', 'opponent_PHX',
           'opponent_SAS', 'opponent_DAL', 'opponent_UTA', 'opponent_MIN',
           'opponent_LAC', 'opponent_MEM', 'action_first_words_Turnaround',
           'shot_zone_basic_Restricted Area', 'opponent_ORL', 'opponent_BOS',
           'combined_shot_type_Layup', 'opponent_TOR', 'opponent_MIA',
           'opponent_SEA', 'opponent_NYK', 'opponent_PHI',
           'shot_zone_basic_In The Paint (Non-RA)', 'opponent_CHA', 'opponent_DET',
           'action_first_words_Pullup', 'opponent_CLE', 'opponent_IND',
           'opponent_WAS', 'opponent_MIL', 'opponent_NJN', 'opponent_CHI',
           'opponent_ATL', 'opponent_NOH', 'action_first_words_Fadeaway',
           'shot_zone_basic_Mid-Range'],
          dtype='object')




```python
X_train = X_train[top50features]

X_test_1 = X_test_1[top50features]
X_test_2 = X_test_2[top50features]
X_test_3 = X_test_3[top50features]
X_test_4 = X_test_4[top50features]
X_test_5 = X_test_5[top50features]
```

### 用線性回歸作預測


```python
regressor = LinearRegression()

regressor.fit(X_train,Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
scores = cross_validation.cross_val_score(regressor, X_train, Y_train, cv=10)
print ("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() / 2))    
```

    Accuracy: 0.123 (+/- 0.007)


### 開始test

#### 第一題- 1997年季後賽Game5 @UTAH，延長賽剩7秒，落後3分的湖人隊，是否能靠老大一舉追平呢？


```python
display(Image(url='https://i.imgur.com/K6PQviM.gif'))
```


<img src="https://i.imgur.com/K6PQviM.gif"/>



```python
np.round(regressor.predict(X_test_1))
```




    array([[0.]])



* 預測結果Kobe這球不會進


```python
display(Image(url='https://i.imgur.com/YoOHeO2.gif'))
```


<img src="https://i.imgur.com/YoOHeO2.gif"/>


* 結果真的沒進，還airball，哈哈哈(1/1)

#### 第二題- 2001季後賽Game1 @SAS，第三節剩57秒，Kobe使出了苦練多年的dream shake，將進球與否？


```python
display(Image(url='https://i.imgur.com/q7mcHZr.gif'))
```


<img src="https://i.imgur.com/q7mcHZr.gif"/>



```python
np.round(regressor.predict(X_test_2))
```




    array([[1.]])



* 預測結果老大帥氣得分


```python
display(Image(url='https://i.imgur.com/whyUbuw.gif'))
```


<img src="https://i.imgur.com/whyUbuw.gif"/>


* 老大果然沒讓球迷失望，帥氣入網，真的帥(2/2)

#### 第三題- 2006季後賽Game4 v.s. PHX，延長賽最後出手，是否一雪前恥呢？


```python
display(Image(url='https://i.imgur.com/9c2SVXX.gif'))
```


<img src="https://i.imgur.com/9c2SVXX.gif"/>



```python
np.round(regressor.predict(X_test_3))
```




    array([[0.]])



* 保守預測這球不進


```python
display(Image(url='https://i.imgur.com/TVEdivF.gif'))
```


<img src="https://i.imgur.com/TVEdivF.gif"/>


* 結果這球決殺了太陽隊，贏得勝利，更贏得球迷的芳心<3，但預測錯誤了QQ(2/3)

#### 第四題- 2010例行賽@BOS，湖人又落後一分，老大是否能利用他的無條件蝦捲跳投，擊敗宿敵呢？


```python
display(Image(url='https://i.imgur.com/AiayiFK.gif'))
```


<img src="https://i.imgur.com/AiayiFK.gif"/>



```python
np.round(regressor.predict(X_test_4))
```




    array([[1.]])



* 經過上一題的愚昧，預測結果就不再小看老大


```python
display(Image(url='https://i.imgur.com/H346e7f.gif'))
```


<img src="https://i.imgur.com/H346e7f.gif"/>


* 老大果然射進啦！守多緊都沒用啦！(3/4)

#### 第五題- 2016例行賽v.s. UTAH，老大在他的farewell game，已攻下53分，能否再利用這球刷到56分並將差距縮小至一分？


```python
display(Image(url='https://i.imgur.com/Cji3dAY.gif'))
```


<img src="https://i.imgur.com/Cji3dAY.gif"/>



```python
np.round(regressor.predict(X_test_5))
```




    array([[1.]])



* #相信老大


```python
display(Image(url='https://i.imgur.com/6tLAylj.gif'))
```


<img src="https://i.imgur.com/6tLAylj.gif"/>


* 啊啊啊進去啦～眼眶微濕，你怎能不愛Kobe？(4/5)
