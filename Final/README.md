巨量資料導論final project
=
# Project題目名稱：
Kobe Bryant多會射？- Kobe生涯投籃表現分析與模型預測</br></br>
![img](https://i.imgur.com/Z8DhgsP.jpg)</br>

## 組員：
M064030016張譯方、M054111049沈昌憲、N054030013林昌碩

### 動機：
身為關注國內外籃壇各賽事的小粉絲，一定都對湖人隊傳奇球星Kobe Bryant不陌生，但就連我阿罵都知道他喜歡射籃，卻常常射不進。我們就利用網路資源，找到他生涯精美的投籃資料，並分析之，來檢討Kobe「汝亦知射乎？」

### 計畫摘要：
我們利用資料中的投射型態、投射角度和對手型態等，來進行資料分析及視覺化，看布萊恩先生擅長的投射型態及位置。並切割樣本，來訓練模型，並依據抽取Kobe生涯的5顆射籃，測試訓練結果。

### 研究步驟：
know our data → 視覺化 → 利用其特性重新定義資料 → 切割訓練集、測試集 → 第一次選取features → 用random forest classification選出前50重要的features → 最後用linear regression訓練 → 看是否成功預測Kobe的那5顆射籃

### 參考資料：
1. http://adataanalyst.com/kaggle/kaggle-tutorial-kobe-bryant/
2. http://gymining.blogspot.com/2016/05/kobe-bryant-shot-selection.html
