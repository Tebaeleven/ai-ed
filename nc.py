import math
import matplotlib.pyplot as plt

class NeuronUnit:
    def __init__(self, name: str, x) -> None:
        # プロパティ
        self.name = name 
        self.x = x
        self.hidden_weights = [0.2,-0.5,-0.1,0.5]
        self.output_weights = [0.8,-0.2]

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
        max_weight = 2  # 重みの上限・下限を設定

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
    target = 0

    # ニューロン1段目
    unit1 = NeuronUnit(name="unit1", x=x1)
    unit2 = NeuronUnit(name="unit2", x=unit1.y)

    
    # エラーを保存するリスト
    errors = []

    # 出力を保存するリスト
    outputs = []

    # トレーニングを10回繰り返す

    # unit1と2を使った場合
    for i in range(200):
        unit1.forward()
        # unit1 の出力を unit2 の入力に設定し、unit2 の順伝播
        unit2.x = unit1.y
        unit2.forward()

        # unit2 のエラーを計算し、全体で使用
        error = unit2.calculate_error(target)  # ターゲットは1
        errors.append(error)
        outputs.append(unit2.y)  # 出力を保存

        # unit2 を誤差 error を引数にして学習
        unit2.train(error)
        unit1.train(error)

    # # unit1のみを使った場合
    # for i in range(10):
    #     unit1.forward()
    #     error = unit1.calculate_error(1)  # ターゲットは1
    #     errors.append(error)
    #     outputs.append(unit1.y)  # 出力を保存
    #     unit1.train(error)

    # エラーと出力のグラフを同じウィンドウに描画
    plt.figure(figsize=(10, 5))

    # エラーのグラフ
    plt.subplot(1, 2, 1)
    plt.plot(errors, label='Error over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error over Training Iterations')
    plt.legend()

    # 出力のグラフ
    plt.subplot(1, 2, 2)
    plt.plot(outputs, label='Output over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Output')
    plt.title('Output over Training Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    

if __name__ == "__main__":
    main()
