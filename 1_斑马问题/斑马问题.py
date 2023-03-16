from kanren import run, eq, membero, var, conde  # kanren一个描述性Python逻辑编程系统
from kanren.core import lall  # lall包用于定义规则
import time


def left(x,y,units):
    groups=zip(units,units[1:])
    return membero((x,y),groups)

def right(x,y,units):
    return left(y,x,units)

def next(x,y,units):
    return conde([left(x, y, units)], [right(x, y, units)])



class Agent:
    """
    推理智能体.
    """

    def __init__(self):
        """
        智能体初始化.
        """

        self.units = var()  # 单个unit变量指代一座房子的信息(国家，工作，饮料，宠物，颜色)
        # 例如('英国人', '油漆工', '茶', '狗', '红色')即为正确格式，但不是本题答案
        # 请基于给定的逻辑提示求解五条正确的答案
        self.rules_zebraproblem = None  # 用lall包定义逻辑规则
        self.solutions = None  # 存储结果

    def define_rules(self):
        """
        定义逻辑规则.
        """

        self.rules_zebraproblem = lall(
            (eq, (var(), var(), var(), var(), var()), self.units),
            # self.units共包含五个unit成员，即每一个unit对应的var都指代一座房子(国家，工作，饮料，宠物，颜色)
            # 各个unit房子又包含五个成员属性: (国家，工作，饮料，宠物，颜色)
            

            # 示例：基于问题信息可以提炼出，有人养斑马，有人喜欢和矿泉水等信息
            (membero, (var(), var(), var(), '斑马', var()), self.units),
            (membero, (var(), var(), '矿泉水', var(), var()), self.units),
            (membero, ('英国人',var(),var(),var(),'红色'),self.units),
            (membero, ('西班牙人', var(), var(), '狗', var()), self.units),
            (membero, ('日本人', '油漆工', var(), var(), var()), self.units),
            (membero, ('意大利人', var(), '茶', var(), var()), self.units),
            (eq, (('挪威人', var(), var(), var(), var()) ,var(),var(),var(),var()), self.units),
            (right, (var(), var(), var(), var(), '绿色'), (var(), var(), var(), var(), '白色'), self.units),
            (membero, (var(), '摄影师', var(), '蜗牛', var()), self.units),
            (membero, (var(), '外交官', var(), var(), '黄色'), self.units),
            (eq, ( var(),var(), (var(), var(), '牛奶', var(), var()), var(),var() ),self.units),
            (membero, (var(), var(), '咖啡', var(), '绿色'), self.units),
            (next, ('挪威人', var(), var(), var(), var()), (var(), var(), var(), var(), '蓝色'), self.units),
            (membero, (var(), '小提琴家', '橘子汁', var(), var()), self.units),
            (next, (var(), var(), var(), '狐狸', var()),(var(), '医生', var(), var(), var()),self.units),
            (next, (var(), var(), var(), '马', var()), (var(), '外交官', var(), var(), var()), self.units)
        )



    def solve(self):
        """
        规则求解器(请勿修改此函数).
        return: 斑马规则求解器给出的答案，共包含五条匹配信息，解唯一.
        """

        self.define_rules()
        self.solutions = run(0, self.units, self.rules_zebraproblem)
        return self.solutions


agent = Agent()
solutions = agent.solve()

# 提取解释器的输出
output = [house for house in solutions[0] if '斑马' in house][0][4]
print ('\n{}房子里的人养斑马'.format(output))
output = [house for house in solutions[0] if '矿泉水' in house][0][4]
print ('{}房子里的人喜欢喝矿泉水'.format(output))

# 解释器的输出结果展示
for i in solutions[0]:
    print(i)