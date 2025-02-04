import math
import matplotlib.pyplot as plt
import numpy as np

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
        max_weight = 1  # 重みの上限・下限を設定

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
    # XORの入力データと出力データ
    input_datas = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    output_datas = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1]
    ])

    # エラーを保存するリスト
    errors = []

    # 出力を保存するリスト
    outputs = []

    # トレーニングを1000回繰り返す
    for epoch in range(100):
        total_error = 0
        epoch_outputs = []  # 各エポックの出力を保存するリスト
        for input_data, output_data in zip(input_datas, output_datas):
            x1, x2 = input_data
            target1, target2 = output_data

            # ニューロン1段目
            unit1_top = NeuronUnit(name="unit1", x=x1)
            unit2_top = NeuronUnit(name="unit2", x=unit1_top.y)

            # ニューロン2段目
            unit3_bottom = NeuronUnit(name="unit3", x=x2)
            unit4_bottom = NeuronUnit(name="unit4", x=unit3_bottom.y)


            # unit1の出力をunit2,unit4の入力にする
            # unit1とunit2の順伝播
            unit1_top.forward()

            unit2_top.x = unit1_top.y
            unit4_bottom.x = unit1_top.y

            # unit3の出力をunit4,unit2の入力にする
            unit3_bottom.forward()

            unit4_bottom.x = unit3_bottom.y
            unit2_top.x = unit3_bottom.y

            unit2_top.forward()
            unit4_bottom.forward()

            # unit2のエラーを計算し、unit1とunit2を学習
            error1 = unit2_top.calculate_error(target1)
            unit2_top.train(error1)
            unit1_top.train(error1)

            # unit4のエラーを計算し、unit3とunit4を学習
            error2 = unit4_bottom.calculate_error(target2)
            unit4_bottom.train(error2)
            unit3_bottom.train(error2)

            # エラーを集計
            total_error += abs(error1) + abs(error2)
            epoch_outputs.append((unit2_top.y, unit4_bottom.y))  # 各エポックの出力を保存

        errors.append(total_error)
        outputs.append(epoch_outputs)

        
        # 各エポックの結果を出力
        print(f"Epoch {epoch + 1}: Total Error = {total_error}")
        for i, (output1, output2) in enumerate(epoch_outputs):
            print(f"  Input: {input_datas[i]}, Predicted Output: ({output1:.2f}, {output2:.2f}), Target: {output_datas[i]}")

    # エラーと出力のグラフを同じウィンドウに描画
    plt.figure(figsize=(10, 5))

    # エラーのグラフ
    plt.subplot(1, 2, 1)
    plt.plot(errors, label='Total Error over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Error')
    plt.title('Total Error over Training Iterations')
    plt.legend()

    # 出力のグラフ
    plt.subplot(1, 2, 2)
    plt.plot([output[0] for output in outputs[-1]], label='Output 1 over iterations')
    plt.plot([output[1] for output in outputs[-1]], label='Output 2 over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Output')
    plt.title('Output over Training Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    

if __name__ == "__main__":
    main()
