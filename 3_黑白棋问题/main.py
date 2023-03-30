# 导入棋盘文件
from board import Board

# 初始化棋盘
board = Board()

# 打印初始化棋盘
board.display()
import copy
import math
import random
class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color

    # 节点类（蒙特卡洛搜索树的节点）
    class Node:
        # 初始化
        def __init__(self, board_state, color, parent=None, action=None):
            self.color = color  # 节点执子方
            self.visit_times = 0  # 节点被访问次数
            self.reward = 0.0  # 当前评分
            self.board_state = board_state  # 当前状态下的棋盘
            self.black_times = 0
            self.white_times = 0
            self.parent = parent  # 父节点
            self.children = []  # 子节点
            self.action = action

        # 节点是否完全扩展
        def is_all_expand(self):
            # 当前棋盘下，color一方可以下棋的合法位置个数
            all_count = len(list(self.board_state.get_legal_actions(self.color)))
            # 已经扩展的节点个数
            children_count = len(self.children)
            if all_count <= children_count:
                return True
            else:
                return False

    # 判断是否还能够扩展树（搜索树是否到底）
    def is_terminal(self, board_state):
        blacks = list(board_state.get_legal_actions('X'))
        whites = list(board_state.get_legal_actions('O'))
        # 当黑棋和白棋都不能再走的时候，说明树已经到底了
        if len(blacks) == 0 and len(whites) == 0:
            return True
        return False

    def expansion(self, node):
        # 还可以走的位置
        a = list(node.board_state.get_legal_actions(node.color))
        # 已经扩展的位置
        b = [i.action for i in node.children]
        # 无路可走时，返回父节点
        if len(a) == 0:
            return node.parent
        # 还可以走的时候，随机找一个可以走的位置action，并保证找的这个action不在已经扩展的范围内
        action = random.choice(a)
        while action in b:
            action = random.choice(a)

        # 将扩展后的情况放到子结点中
        new_state = copy.deepcopy(node.board_state)
        new_state._move(action, node.color)  # 走完这步棋的棋盘状态
        new_color = 'X' if node.color == 'O' else 'O'  # 子节点棋的颜色

        # 为节点添加子节点
        child = self.Node(new_state, new_color, parent=node, action=action)
        node.children.append(child)

        return node.children[-1]

    # 直到达到一个不完全扩展节点（即此节点的仍有未访问的子节点）
    def selection(self, node):
        while not self.is_terminal(node.board_state):
            # 没有可以扩展的节点了，也就是说要模拟这个节点（返回这个节点，直接进入下下步模拟环节）
            if len(list(node.board_state.get_legal_actions(node.color))) == 0:
                return node

            # node还未完全扩展（也就是node的子节点个数<还可以走的位置数）即还有位置没被扩展
            elif not node.is_all_expand():
                # 扩展，即随机返回一个没有扩展的可扩展点，并将其存入children（已扩展的子节点）里面
                new_node = self.expansion(node)
                return new_node
            else:
                # node所有可以扩展的点都被扩展完了，就该选择这个节点的子节点里面ucb最大的点，并让node=这个点，为了继续向下递归选择
                node = self.best_ucb(node)
                return node

    def best_ucb(self, node, ratio=1):
        max_value = -float('inf')  # 预设最大ucb（先设为负无穷）
        max_list = []  # 具有最大ucb的子节点的列表（为了随机返回一个最大ucb的点）
        for c in node.children:
            x = math.sqrt(2.0 * math.log(node.visit_times) / float(c.visit_times))
            if c.color == 'X':
                ucb = c.black_times / (c.black_times + c.white_times) + ratio * x
            else:
                ucb = c.white_times / (c.black_times + c.white_times) + ratio * x
            # 把最大的ucb值赋给max_value，把最大ucb的点加到max_list里
            if ucb == max_value:
                max_list.append(c)
            elif ucb > max_value:
                max_list = [c]
                max_value = ucb

        # 如果子列表没有最大ucb，则返回它的父节点
        if len(max_list) == 0:
            return node.parent
        else:
            # 随机返回一个最大ucb的点
            return random.choice(max_list)

    def stimulation(self, node):
        # 对节点node进行模拟
        new_board = copy.deepcopy(node.board_state)
        new_color = copy.deepcopy(node.color)
        count = 0

        # 当前棋局还可以继续的时候则进入循环
        while not self.is_terminal(new_board):
            actions = list(node.board_state.get_legal_actions(new_color))  # 当前状态下还可以走的点
            if len(actions) == 0:
                # 走到头了
                new_color = 'X' if new_color == 'O' else 'O'
            else:
                # 还可以继续模拟
                action = random.choice(actions)  # 随机模拟
                new_board._move(action, new_color)
                new_color = 'X' if new_color == 'O' else 'O'
            count += 1
            # 最多模拟30次
            if count > 30:
                break
        return new_board.count('X'), new_board.count('O')

    # 反向传播
    def back_propagation(self, node, black, white):
        # 从当前节点（被模拟的节点）开始反向传播，更新他们的父节点
        while (node is not None):
            node.visit_times += 1
            node.black_times += black
            node.white_times += white
            node = node.parent

    def MCTS_search(self, point, maxtimes=100):
        for i in range(maxtimes):
            choice = self.selection(point)
            blackwin_times, whitewin_times = self.stimulation(choice)
            self.back_propagation(choice, black=blackwin_times, white=whitewin_times)

        return self.best_ucb(point).action

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------请实现你的算法代码--------------------------------------

        root_board = copy.deepcopy(board)
        root = self.Node(board_state=root_board, color=self.color)
        action = self.MCTS_search(root)

        # ------------------------------------------------------------------------

        return action
# 导入黑白棋文件
from game import Game  

# 人类玩家黑棋初始化
black_player =  HumanPlayer("X")

# AI 玩家 白棋初始化
white_player = AIPlayer("O")

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)

# 开始下棋
game.run()