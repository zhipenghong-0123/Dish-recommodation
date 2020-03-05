# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
from sklearn.model_selection import train_test_split

# 任务1.1 文件读取
info = pd.read_csv('meal_order_info.csv', encoding='GBK')
detail = pd.read_csv('meal_order_detail.csv')

# 任务1.2 去掉特殊的字符
newdishname = []
for i in detail.dishes_name.tolist():
    if '/n' or '/r' in i:
        a = i.replace('\n', '').replace('\r', '')
        newdishname.append(a)
detail.dishes_name = newdishname

# 任务1.3 构建热度函数
def hot_rating(data):
    return (data - data.min()) / (data.max() - data.min())

# 任务1.4 画图
data = detail[['dishes_name', 'counts', 'amounts']]
data.groupby('dishes_name').sum().sort_values(by='counts').tail(10)['counts'].plot.barh(title="菜品热销Top10")
print("2016年8月销量为：" + str(sum(data.counts * data.amounts)))

# 任务2.1 订单状态占比
info.order_status.value_counts(normalize=True)

# 任务2.2 选取有效的订单数据
info = info[info.order_status == 1]
info = info.drop(
    ['mode', 'cashier_id', 'pc_id', 'order_number', 'print_doc_bill_num', 'lock_table_info', 'check_closed'], axis=1)
detail = detail.drop(
    ['logicprn_name', 'parent_class_name', 'cost', 'discount_amt', 'discount_reason', 'kick_back', 'add_info',
     'bar_code'], axis=1)
detail = detail.drop(index=(detail.loc[(detail['dishes_name'] == '白饭/小碗')].index))
detail = detail.drop(index=(detail.loc[(detail['dishes_name'] == '白饭/大碗')].index))

# 任务2.3 选取主要特征
data2 = detail[['dishes_name', 'emp_id']]

# 任务3.1 划分测试集与训练集
data2.emp_id.value_counts()
dish_order = data2.drop_duplicates()
# 三个菜以上的顾客名单
customer_list = dish_order.emp_id.value_counts()[dish_order.emp_id.value_counts() > 3].index.tolist()
# 获取所有菜品名称
dishes_name = list(set(detail.dishes_name))
# 按照顾客名单划分测试集与训练集
train_data, test_data = train_test_split(customer_list, test_size=0.3)

# 3.2 二维矩阵---函数
def two_dimension(customer_list, dishes_list, df):
    customer_dish = pd.DataFrame(index=customer_list, columns=dishes_list)
    for i in customer_list:
        for j in df[df.emp_id == i].dishes_name.tolist():
            customer_dish.loc[i, j] = 1
    customer_dish = customer_dish.fillna(0)
    return customer_dish


#构建训练集和测试集二维矩阵

train_customer_dish = two_dimension(train_data, dishes_name, dish_order)
test_customer_dish = two_dimension(test_data, dishes_name, dish_order)

# 任务4 模型构建
class ItemCF:
    def __init__(self, train_TwoMatrix, test_TwoMatrix):
        print('-----------模型实例化-----------')
        self.train_TwoMatrix = train_TwoMatrix
        self.test_TwoMatrix = test_TwoMatrix
        self.train_customerID = self.train_TwoMatrix.index.tolist()
        self.test_customerID = self.test_TwoMatrix.index.tolist()
        self.dish = train_TwoMatrix.columns.tolist()
        print('----------生成相似度矩阵---------')
        self.sim_all()
        print('---------相似度矩阵生成完毕-------')
        print('---------生成全顾客推荐列表-------')
        self.all_customer_recommodation_list()
        print('----------全顾客推荐列表生成完毕---------')

    # 4.1 相似度
    # 欧几里得距离
    def sim_two(self, dish1: list, dish2: list):
        return 1 / (1 + np.sqrt(((np.array(dish1) - np.array(dish2)) ** 2).sum()))

    def sim_all(self):
        self.sim_matrix = pd.DataFrame(columns=self.dish, index=self.dish)
        for i in range(len(self.sim_matrix.columns)):
            for j in range(len(self.sim_matrix.columns)):
                self.sim_matrix.iloc[i, j] = self.sim_two(self.train_TwoMatrix.iloc[:, i].tolist(),
                                                          self.train_TwoMatrix.iloc[:, j].tolist())
        return self.sim_matrix

    # 4.2 生成推荐键对列表
    def recommodation_dic(self, customerName, df):
        Name_rec_dic = {}
        for i in self.dish:
            sim = 0
            for j in self.dish:
                sim += self.sim_matrix.loc[i, j] * df.loc[customerName, j]
            Name_rec_dic[i] = sim
        Name_rec_dic = sorted(Name_rec_dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        return Name_rec_dic

    # 生成全顾客推荐字典
    def all_customer_recommodation_list(self):
        self.train_all_dic = {}
        self.test_all_dic = {}
        for i in self.train_customerID:
            self.train_all_dic[i] = list(dict(self.recommodation_dic(i, self.train_TwoMatrix)))
        for i in self.test_customerID:
            self.test_all_dic[i] = list(dict(self.recommodation_dic(i, self.test_TwoMatrix)))
        return self.test_all_dic

    # 生成指定个数的推荐菜品字典
    def select_recommodation_dic(self, n=10):
        self.select_dic = {}
        for i in self.test_all_dic.keys():
            self.select_dic[i] = self.test_all_dic[i][0:n]
        return self.select_dic



# 模型实例化
if __name__ == '__main__':
    model = ItemCF(train_customer_dish, test_customer_dish)

# 5.1构建客户IP字典
customerIP = {}
for i in test_customer_dish.index.tolist():
    customerIP[i] = list(test_customer_dish.loc[i, :][test_customer_dish.loc[i, :] == 1].index.tolist())

# 5.2模型评价

def precise(test_predict_dic, dic):
    all_num = 0
    precise = 0
    total = 0
    for i in list(test_predict_dic.keys()):
        total += len(test_predict_dic[i])
        for j in test_predict_dic[i]:
            if j in dic[i]:
                precise += 1
    for i in list(dic.keys()):
        all_num += len(dic[i])
    print('-------------模型效果分析---------------')
    print('客户总喜好个数为', all_num)
    print('总推荐个数为', total)
    print('准确预测的个数为', precise)
    print('预测准确率为', precise / total)
    print('召回率', precise / all_num)
    return precise / total, precise / all_num


# 计算准确率和召回率

pre = []
rec = []
for i in range(1, 100):
    a, b = precise(model.select_recommodation_dic(i), customerIP)
    pre.append(a)
    rec.append(b)

# 画图分析准确率和召回率
plt.show()
x = list(range(1, 100))
plt.plot(x, pre, color='red')
plt.plot(x, rec, color='green')
plt.legend(['precise', 'recall'])
plt.xlabel('推荐个数n')
plt.savefig('模型评估.png')
plt.show()



