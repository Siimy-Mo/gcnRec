import pickle
import os
import numpy as np
import pandas as pd

# auctionid;bid;bidtime;bidder;bidderrate;openbid;price;item;auction_type

dictItems={}
userCount={}
col_names = ["item", "bid","timestamp",'user','userRate','openPrice','finalPrice','type','duration']
df = pd.read_csv("./data/ebay/biddingRecord.csv", sep=';', header=None, names=col_names, engine='python')
# 类型转换
for col in ( "user","item"):
    df[col] = df[col].astype(np.int32)
for col in ("timestamp", "bid","openPrice","finalPrice"):
    df[col] = df[col].astype(np.float32)

pidList = list(set(df["item"].values.tolist()))
USER_NUM = len(set(df["user"].values.tolist()))
ITEM_NUM = len(pidList)


MAXPRICE = 0
for i in list(pidList): # 以auction为单位
  pid = pidList[i]
  dictItems[pid]={'alluidList':list(),'bidsList':list(),'user1':list(),'user2':list(),'pos_uid':list(),'neg_uid':list()} ### gcn的點 u1 -> u2

  bidRecord = df.loc[df['item'] == pid] #查询对应的item bid records,競拍價格從低到高
  bidRecord = bidRecord.sort_values(by='bid',ascending=True).reset_index()
  finalPrice = bidRecord.iloc[-1]['finalPrice'].tolist()
  if finalPrice> MAXPRICE:
    MAXPRICE = float(finalPrice)

  for index, row in bidRecord.iterrows():
    if row['user'] in userCount:  ###GCN:统计user在所有auction中出现的次数
      userCount[row['user']] += 1
    else:
      userCount[row['user']] = 1

    if index == 0 : ### 這裡設置的初始節點為第一個用戶的自循環，價格是從open price 到當前bid
      lastUser = row['user']

    dictItems[pid]['alluidList'].append(row['user'])
    dictItems[pid]['bidsList'].append(row['bid'])
    dictItems[pid]['user1'].append(lastUser)
    dictItems[pid]['user2'].append(row['user'])
    lastUser = row['user']
    if float(row['finalPrice']) == float(row['bid']): # final user
      dictItems[pid]['pos_uid'].append(lastUser)

  ## novelty。在[U1,U2]中，添加所有alluidList 对 pos_uid的链接。权重未设置
  _uidList = set(dictItems[pid]['alluidList'])
  _uidList.remove(dictItems[pid]['pos_uid'][0])
  for i in range(len(dictItems[pid]['user2'])):
    U1 = dictItems[pid]['user1'][i]
    U2 = dictItems[pid]['user2'][i]
    if U2 == dictItems[pid]['pos_uid'][0] and U1 in _uidList:
      _uidList.remove(U1)

  # 连接未链接的 [U1,U2]
  _uidList = list(_uidList)
  if len(_uidList) !=0:

    for j in range(len(_uidList)):
      dictItems[pid]['user1'].append(_uidList[j])
      dictItems[pid]['user2'].append(dictItems[pid]['pos_uid'][0])

  if(len(dictItems[pid]['alluidList'])>0): # 拥有两个以上的竞拍记录
    for i in range(USER_NUM):
      if i not in dictItems[pid]['alluidList']  :        # 添加neg user,which not exit on bidder list
        dictItems[pid]['neg_uid'].append(i)####111
###GCN:筛选userCount >= 5的用户
###GCN:筛选 >= 5的用户
# for i in list(pidList):

pidCanTest = []
maxbidlen = 0
for itemid in dictItems: 
  bidderCount = len(set(dictItems[itemid]['alluidList']))
  if bidderCount > 1:    # 寻找max bid len，用于填充
    pidCanTest.append(itemid)

  if len(dictItems[itemid]['bidsList']) > maxbidlen:    # 寻找max bid len，用于填充
    maxbidlen = len(dictItems[itemid]['bidsList'])

print('\nThe length of auctions who have more than 2 pos sample(Test): ', len(pidCanTest))#479
print('\nThe maxmium length of bids record: ', maxbidlen)

trainNum = int(len(pidCanTest)* 0.8)
trainpid = pidCanTest[:trainNum] # 随机取样用于train的物品,585场中有468个train 117 test
testpid = list(set(pidCanTest) - set(trainpid))
print('train/ test item number: ',len(trainpid), len(testpid))# 383 96

data=[]
if not os.path.exists(data):
  os.makedirs('data')
# pickle.dump( data , open('data/train.txt','wb'))
# pickle.dump( data , open('data/test.txt','wb'))
# pickle.dump( data , open('data/all_train_seq.txt','wb'))

print('Done!')