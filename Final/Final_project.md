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
df.head()
```

```python
df.info()
```

```python
df.describe()
```

```python
df.describe(include=['object', 'category'])
```

### Data視覺化

```python
from IPython.display import Image
from IPython.display import display
Image('https://i.imgur.com/IykGbhG.jpg')
```

根據上圖可以看出我們布萊恩先生最愛的射籃型態是jump shot，但其命中率只有39.1%；而扣除掉dunk，布萊恩先生命中率最高的是bank
shot，但只佔他出手比率的0.5，%顯然不喜歡這種投射姿勢><

```python
Image('https://i.imgur.com/r6PRwR3.jpg')
```

根據此圖可看出老大較喜歡在球場左邊進行攻擊，或許這和他慣用手為右手有關。而且儘管老大較常在左邊做出手，他的命中率也不遜於右側，所以下次和Kobe在球場上較勁時，記得踩住他右切的路線，不然會被他射得不要不要的。<br/>
再來是老大統治NBA的時代，三分線瘋狂出手及過了半場旱地拔蔥Logo shot尚未盛行，因此他在三分線外的命中率也不算太高。

```python
Image('https://i.imgur.com/jdjEJcM.jpg', width="400")
```

根據此圖，老大的投射命中率隨投射距離明顯地遞減，在16ft.-24ft.間他瘋狂射籃，射了將近7000發，但命中率僅40.2%，而印象派球迷也常詬病他中距離的出手選擇，以實際數據分析結果也是相同結果。

```python
Image('https://i.imgur.com/SDiLPM8.jpg')
```

在2008-09賽季Kobe老大得到了NBA
MVP的殊榮，當年他幹進了生涯第二高的1864分，2分球命中率49.6%、3分球命中率為35.5%，效率頗高，可說是魅力四射、風靡萬眾的一年<br/>
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

```python
features_good_fit = pd.DataFrame(rfc.feature_importances_, index=X_train.columns, columns=["good_fit"])
top50features = features_good_fit.sort_values("good_fit", ascending=False).head(50).index
```

```python
top50features
```

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

```python
scores = cross_validation.cross_val_score(regressor, X_train, Y_train, cv=10)
print ("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() / 2))    
```

### 開始test

#### 第一題- 1997年季後賽Game5 @UTAH，延長賽剩7秒，落後3分的湖人隊，是否能靠老大一舉追平呢？

```python
display(Image(url='https://i.imgur.com/K6PQviM.gif'))
```

```python
np.round(regressor.predict(X_test_1))
```

* 預測結果Kobe這球不會進

```python
display(Image(url='https://i.imgur.com/YoOHeO2.gif'))
```

* 結果真的沒進，還airball，哈哈哈(1/1)

#### 第二題- 2001季後賽Game1 @SAS，第三節剩57秒，Kobe使出了苦練多年的dream shake，將進球與否？

```python
display(Image(url='https://i.imgur.com/q7mcHZr.gif'))
```

```python
np.round(regressor.predict(X_test_2))
```

* 預測結果老大帥氣得分

```python
display(Image(url='https://i.imgur.com/whyUbuw.gif'))
```

* 老大果然沒讓球迷失望，帥氣入網，真的帥(2/2)

#### 第三題- 2006季後賽Game4 v.s. PHX，延長賽最後出手，是否一雪前恥呢？

```python
display(Image(url='https://i.imgur.com/9c2SVXX.gif'))
```

```python
np.round(regressor.predict(X_test_3))
```

* 保守預測這球不進

```python
display(Image(url='https://i.imgur.com/TVEdivF.gif'))
```

* 結果這球決殺了太陽隊，贏得勝利，更贏得球迷的芳心<3，但預測錯誤了QQ(2/3)

#### 第四題- 2010例行賽@BOS，湖人又落後一分，老大是否能利用他的無條件蝦捲跳投，擊敗宿敵呢？

```python
display(Image(url='https://i.imgur.com/AiayiFK.gif'))
```

```python
np.round(regressor.predict(X_test_4))
```

* 經過上一題的愚昧，預測結果就不再小看老大

```python
display(Image(url='https://i.imgur.com/H346e7f.gif'))
```

* 老大果然射進啦！守多緊都沒用啦！(3/4)

#### 第五題- 2016例行賽v.s. UTAH，老大在他的farewell game，已攻下53分，能否再利用這球刷到56分並將差距縮小至一分？

```python
display(Image(url='https://i.imgur.com/Cji3dAY.gif'))
```

```python
np.round(regressor.predict(X_test_5))
```

* #相信老大

```python
display(Image(url='https://i.imgur.com/6tLAylj.gif'))
```

* 啊啊啊進去啦～眼眶微濕，你怎能不愛Kobe？(4/5)
