import networkx as nx
import numpy as np
import pandas as pd

# 一些简单的应用函数

def get_Features(USER_NUM,ITEM_NUM):
    UserFeatures=np.identity(USER_NUM)
    ItemFeatures=[]

    if(True):
      UsrDat=get_UserData() # 只是bidder rate 1
      UserFeatures=np.concatenate((UserFeatures,UsrDat), axis=1) 
      UserFeatures =UserFeatures.astype(np.float32)

    if(True):
      ItemFeatures=get_ItemData()
      ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), ItemFeatures), axis=1) 
    return UserFeatures,ItemFeatures

def get_UserData():
    df = pd.read_csv('./data/ebay/DealerFeatures.csv', sep=';', engine='python')
    df=df.sort_values(by=['bidder'])
    del df['bidder']
    values =df.values.astype(np.float32)
    return values

def get_ItemData():
    df = pd.read_csv('./data/ebay/ItemFeatures.csv', sep=';', engine='python')
    df=df.sort_values(by=['auctionid'])
    del df['auctionid']
    df=pd.concat([df,df['item'].str.get_dummies(sep=' ').add_prefix('Name_').astype('int8')],axis=1)      #  Cartier wristwatch -> 0 0
    df=pd.concat([df,df['auction_type'].str.get_dummies(sep=' ').add_prefix('Auction_').astype('int8')],axis=1) 
    del df['item']
    del df['auction_type']
    del df['Auction_auction'],df['Auction_day']

    df=pd.get_dummies(df,dummy_na=True)
    df=df.fillna(df.mean())
    df=df.dropna(axis=1, how='all')

    values=df.values
    return values

def generate_gcnData(USER_NUM, trainRatio = 0.8):
    dictItems={}
  # 录入item - final user 的关系：
  # auctionid;bid;bidtime;bidder;bidderrate;openbid;price;item;auction_type
    col_names = ["item", "bid","timestamp",'user','userRate','openPrice','finalPrice','type','duration']
    df = pd.read_csv("./data/ebay/biddingRecord.csv", sep=';', header=None, names=col_names, engine='python')
    pidList = list(set(df["item"].values.tolist()))

    # 类型转换
    for col in ( "user","item"):
        df[col] = df[col].astype(np.int32)
    for col in ("timestamp", "bid","openPrice","finalPrice"):
        df[col] = df[col].astype(np.float32)

    MAXPRICE = 0
    for i in range(len(pidList)):
      pid = pidList[i]
      dictItems[pid]={'alluidList':list(),'bidsList':list(),'user1':list(),'user2':list(),'pos_uid':list(),'neg_uid':list()} ### gcn的點 u1 -> u2

      bidRecord = df.loc[df['item'] == pid] #查询对应的item bid records,競拍價格從低到高
      bidRecord = bidRecord.sort_values(by='bid',ascending=True).reset_index()
      finalPrice = bidRecord.iloc[-1]['finalPrice'].tolist()
      if finalPrice> MAXPRICE:
        MAXPRICE = float(finalPrice)

      for index, row in bidRecord.iterrows():
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
    
    trainNum = int(len(pidCanTest)* trainRatio)
    trainpid = pidCanTest[:trainNum] # 随机取样用于train的物品,585场中有468个train 117 test
    testpid = list(set(pidCanTest) - set(trainpid))
    print('train/ test item number: ',len(trainpid), len(testpid))# 383 96

    return dictItems, MAXPRICE, maxbidlen,  [trainpid,testpid]