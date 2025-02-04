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
        self.h1 = self.relu(self.h1)
        self.h2 = self.relu(self.h2)

        # デバッグ：中間層の結果を表示
        # print(f"x: {self.x} * {self.hidden_weights[0]} + {self.x} * {self.hidden_weights[2]} = {self.h1}")
        # print(f"x: {self.x} * {self.hidden_weights[1]} + {self.x} * {self.hidden_weights[3]} = {self.h2}")

        # 出力層の出力
        self.y = self.h1 * self.output_weights[0] + self.h2 * self.output_weights[1]
        # print(f"y: {self.y}")

    def __str__(self) -> str:
        return f"NeuronUnit(name={self.name}, hidden_weights={self.hidden_weights}, output_weights={self.output_weights})"
    
    def train(self, input, target):
        # 誤差を計算
        error = target - input

        print(f"error: {error}")

        # 誤差を修正

        max_weight = 5

        # もし誤差が0より大きいなら増やす、0より小さいなら減らす
        if error > 0:
            # 重みを増やす

            # 出力層の重み
            self.output_weights[1] += error/2
            self.output_weights[1] = min(self.output_weights[1], max_weight)

            # 中間層の重み
            self.hidden_weights[2] += self.output_weights[0] / 2
            self.hidden_weights[2] = min(self.hidden_weights[2], max_weight)
            self.hidden_weights[3] += self.output_weights[1] / 2
            self.hidden_weights[3] = min(self.hidden_weights[3], max_weight)
        else:
            # 重みを減らす

            # 出力層の重み
            self.output_weights[0] -= error/2
            self.output_weights[0] = max(self.output_weights[0], -max_weight)
            # 中間層の重み
            self.hidden_weights[0] -= self.output_weights[0] / 2
            self.hidden_weights[0] = max(self.hidden_weights[0], -max_weight)
            self.hidden_weights[1] -= self.output_weights[1] / 2
            self.hidden_weights[1] = max(self.hidden_weights[1], -max_weight)
          

    def relu(self, x):
      return max(0, x)
    
    def calculate_error(self, target):
        return target - self.y

def main():
    # 入力
    x1 = 1.5

    # ニューロン1段目
    unit1 = NeuronUnit(name="unit1", x=x1)
    
    # エラーを保存するリスト
    errors = []

    # トレーニングを10回繰り返す
    for _ in range(100):
        unit1.forward()
        error = unit1.calculate_error(1)  # ターゲットは1
        errors.append(error)
        unit1.train(unit1.y, 1)

    # グラフを描画
    plt.plot(errors, label='Error over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error over Training Iterations')
    plt.legend()
    plt.show()

    # unit2 = NeuronUnit(name="unit2", x=unit1.y)
    # unit2.forward()
    # print(unit2.y)

    

    # # ニューロン2段目
    # unit3 = NeuronUnit(name="unit3", x=x2)
    # unit3.forward()
    # print(unit3.y)

    # unit4 = NeuronUnit(name="unit4", x=unit3.y)
    # unit4.forward()
    # print(unit4.y)

    
    
    

if __name__ == "__main__":
    main()
