import math
import matplotlib.pyplot as plt
import numpy as np

class NeuronUnit:
    def __init__(self, name: str, x) -> None:
        # プロパティ
        self.name = name 
        self.x = x
        # self.hidden_weights = [0.2,-0.5,-0.1,0.5]
        # self.output_weights = [0.8,-0.2]

        self.hidden_weights = np.random.randn(4)
        self.output_weights = np.random.randn(2)

        # 中間層の出力
        self.h1 = 0
        self.h2 = 0

        # 出力層の出力
        self.y = 0


    def forward(self):
        # 中間層の出力
        self.h1 = self.x * self.hidden_weights[0] + self.x * self.hidden_weights[2]
        self.h2 = self.x * self.hidden_weights[1] + self.x * self.hidden_weights[3]

        # 活性化関数
        self.h1 = self.sigmoid(self.h1)
        self.h2 = self.sigmoid(self.h2)

        # デバッグ：中間層の結果を表示
        # print(f"x: {self.x} * {self.hidden_weights[0]} + {self.x} * {self.hidden_weights[2]} = {self.h1}")
        # print(f"x: {self.x} * {self.hidden_weights[1]} + {self.x} * {self.hidden_weights[3]} = {self.h2}")

        # 出力層の出力
        self.y = self.h1 * self.output_weights[0] + self.h2 * self.output_weights[1]
        self.y = self.sigmoid(self.y)
        # print(f"y: {self.y}")

    def __str__(self) -> str:
        return f"NeuronUnit(name={self.name}, hidden_weights={self.hidden_weights}, output_weights={self.output_weights})"
    
    def train(self, error):
        """
        引数 error を用いて各重みを調整する。
        ここでは単純に誤差が正の場合は一部の重みを増やし、
        負の場合は別の重みを減らす例としています。
        """
        print(f"[{self.name}] error: {error}")
        max_weight = 10  # 重みの上限・下限を設定

        if error > 0:
            # 重みを増やす

            # 出力層の重み
            self.output_weights[1] += error/2

            # 中間層の重み
            self.hidden_weights[2] += self.output_weights[0] / 2
            self.hidden_weights[3] += self.output_weights[1] / 2
            self.hidden_weights[3] = min(self.hidden_weights[3], max_weight)
            self.hidden_weights[2] = min(self.hidden_weights[2], max_weight)
            self.output_weights[1] = min(self.output_weights[1], max_weight)

        else:
            # 重みを減らす

            # 出力層の重み
            self.output_weights[0] -= error/2
            # 中間層の重み
            self.hidden_weights[0] -= self.output_weights[0] / 2
            self.hidden_weights[1] -= self.output_weights[1] / 2


            self.hidden_weights[1] = max(self.hidden_weights[1], -max_weight)
            self.hidden_weights[0] = max(self.hidden_weights[0], -max_weight)
            self.output_weights[0] = max(self.output_weights[0], -max_weight)
          

    def relu(self, x):
      return max(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def calculate_error(self, target):
        return target - self.y
    
    def step(self, x):
        return 1 if x > 0 else 0

def main():
    # 入力
    x1 = 1
    x2 = 1
    target = 1 # 0
    target2 = 0 # 1

    # ニューロン1段目
    unit1 = NeuronUnit(name="unit1", x=x1)
    unit2 = NeuronUnit(name="unit2", x=0)

    # ニューロン2段目
    unit3 = NeuronUnit(name="unit3", x=0)
    unit4 = NeuronUnit(name="unit4", x=0)

    
    # エラーを保存するリスト
    errors = []

    # unit1と2を使った場合
    outputs_unit2 = []  # unit2の出力を保存するリスト
    outputs_unit4 = []  # unit4の出力を保存するリスト

    for i in range(500):
        unit1.x = x1
        unit3.x = x2
        
        # 一層目の計算
        unit1.forward()
        unit3.forward()

        # 2層目の計算
        # 2には1,3の出力を足したものを入れる
        unit2.x= unit1.y + unit3.y
        
        # 4には1,3の出力を足したものを入れる
        unit4.x = unit1.y + unit3.y

        unit2.forward()
        unit4.forward()

        # 2のエラーを計算
        error_unit2 = unit2.calculate_error(target)
        errors.append(error_unit2)
        outputs_unit2.append(unit2.y)  # unit2の出力を保存
        unit2.train(error_unit2)
        unit1.train(error_unit2)

        # 4のエラーを計算
        error_unit4 = unit4.calculate_error(target2)
        errors.append(error_unit4)
        outputs_unit4.append(unit4.y)  # unit4の出力を保存
        unit4.train(error_unit4)
        unit3.train(error_unit4)

    unit1.x = 1
    unit3.x = 0
    unit1.forward()
    unit3.forward()
    
    unit2.x = unit1.y + unit3.y
    unit4.x = unit1.y + unit3.y

    unit2.forward()
    unit4.forward()

    print(f"input ({unit1.x},{unit3.x}) -> output: {unit2.y}, {unit4.y}")

    # エラーと出力のグラフを同じウィンドウに描画
    plt.figure(figsize=(15, 5))

    # エラーのグラフ
    plt.subplot(1, 3, 1)
    plt.plot(errors, label='Error over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error over Training Iterations')
    plt.legend()

    # unit2の出力のグラフ
    plt.subplot(1, 3, 2)
    plt.plot(outputs_unit2, label='Unit2 Output over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Output')
    plt.title('Unit2 Output over Training Iterations')
    plt.legend()

    # unit4の出力のグラフ
    plt.subplot(1, 3, 3)
    plt.plot(outputs_unit4, label='Unit4 Output over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Output')
    plt.title('Unit4 Output over Training Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    
    

if __name__ == "__main__":
    main()
