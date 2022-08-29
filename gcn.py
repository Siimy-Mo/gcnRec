import argparse
import pickle
import time
from utils import generate_gcnData, get_Features
from model import *

DIM=60
USER_NUM = 892
ITEM_NUM = 575
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)

def main():
    UserFeatures,ItemFeatures=get_Features(USER_NUM,ITEM_NUM)

    print(UserFeatures.shape) # id的熱編碼or id+features _____892 893
    print(ItemFeatures.shape) # 【openbid-type-duration】 _____575 588

    # 初始化model
    # model = trans_to_cuda(SessionGraph(opt, n_node))
    in_size = UserFeatures.shape[1]
    out_size = 100
    model = GCN(in_size,16,out_size)
    
    # 初始化各项计数变量
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        # 将train 和test 投入model中
        # hit, mrr = train_test(model, train_data, test_data)

        # 获得结果，对比best
        print('Best Result:')

    # display result：
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    # 读取数据
    dictItems,MAXPRICE,MAXBIDS,train_Test_pidList = generate_gcnData(USER_NUM)

    #投入main中
    main(train_Test_pidList)