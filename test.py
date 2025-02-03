import math
import random
import matplotlib.pyplot as plt
from typing import Literal

# シグモイド関数


def sigmoid(x, u0=0.4):
    return 1 / (1 + math.exp(-2 * x / u0))

# シグモイド関数の微分


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Neuron:
    def __init__(
        self,
        input_neurons: list["Neuron"],
        type: Literal["p", "n"],
        alpha: float = 0.8,
        activation=sigmoid,
        activation_derivative=sigmoid_derivative,
    ) -> None:
        self.type = type
        self.activation = activation
        self.alpha = alpha
        self.activation_derivative = activation_derivative
        # 重みの初期化
        self.weights = []

        # 重みを正の値にするか負の値にするかを決定

        # + と + の場合は1
        # + と - の場合は-1
        # - と + の場合は-1
        # - と - の場合は1
        for n in input_neurons:
            if type == "p":
                if n.type == "p":
                    ope = 1
                else:
                    ope = -1
            else:
                if n.type == "p":
                    ope = -1
                else:
                    ope = 1
            self.weights.append(random.random() * ope)

        # オペレーター
        # 自分がpなら1、nなら-1
        self.operator = 1 if type == "p" else -1

        # 重みが1か-1かを保存
        self.operator_weights = [n.operator for n in input_neurons]

        # upper,lowerのリスト
        self.upper_idx_list = []
        self.lower_idx_list = []

        # 入力ニューロンがpならupper_idx_listに、nならlower_idx_listに追加
        for i, n in enumerate(input_neurons):
            if n.type == "p":
                self.upper_idx_list.append(i)
            else:
                self.lower_idx_list.append(i)

    def forward(self,x):
        # 入力の長さと重みの長さが同じか確認
        assert len(self.weights) == len(x)

        # 重みと入力の内積
        # 入力＊重み
        y = [x[i] * self.weights[i] for i in range(len(self.weights))]
        y = sum(y)
        self.prev_in = x  # update用に一時保存
        self.prev_out = y  # update用に一時保存
        y = self.activation(y)
        return y

    def update_weight(self, delta_out, direct: str):

        # 誤差拡散法

        grad = self.activation_derivative(abs(self.prev_out))

        # directが"upper"の場合、upper_idx_listを使用
        if direct == "upper":
            indices = self.upper_idx_list
        # directが"lower"の場合、lower_idx_listを使用
        else:
            indices = self.lower_idx_list

        # 重みを更新

        # 増やす用、減らす用のindeciesを回す
        for idx in indices:
            
            # 重みの更新量を計算する
            delta = self.alpha * self.prev_in[idx] # 学習率
            delta *= grad # 勾配
            delta *= delta_out * self.operator * self.operator_weights[idx] # 誤差拡散法
            
            self.weights[idx] += delta # 実際に重みを更新

    def __str__(self):
        return f"Neuron(type={self.type}, weights={self.weights})"
        

class ThreeLayerModel:
    def __init__(
        self,
        input_num: int,
        hidden_num: int,
        alpha: float = 0.8,
        beta: float = 0.8,
    ) -> None:
        self.beta = beta

        # バイアス？
        hd_p = Neuron([],"p")
        hd_n = Neuron([],"n")

        # 入力層
        # PとNそれぞれ作成
        
        inputs: list[Neuron] = []

        for n in range(input_num):
            inputs.append(Neuron([],"p"))
            inputs.append(Neuron([],"n"))


        # 中間層
        self.hidden_neurons: list[Neuron] = []

        for i in range(hidden_num):
            self.hidden_neurons.append(
                Neuron(
                    [hd_p, hd_n] + inputs, # バイアスとニューロンの入力を渡す
                    type=("p" if i % 2 == 1 else "n"),  # 元コードに合わせて-から作成
                )
            )

        # 出力層

        # バイアスと中間層のニューロンを接続
        self.out_neuron = Neuron([hd_p, hd_n] + self.hidden_neurons, "p")

    def forward(self, inputs):
        # 入力用の配列、pとnを複製

        # 入力層
        x = []
        for n in inputs:
            x.append(n)
            x.append(n)

        # 中間層
        # バイアスと入力を渡す
        x = [h.forward([self.beta, self.beta] + x) for h in self.hidden_neurons]

        # 出力層
        # バイアスと中間層のニューロンを渡す
        x = self.out_neuron.forward([self.beta, self.beta] + x)

        return x
    
    def train(self, inputs, target):
        x = self.forward(inputs)

        # 誤差を計算
        diff = target - x

        # 誤差が正ならupper、負ならlower
        if diff > 0:
            direct = "upper"
        else:
            direct = "lower"
            diff = -diff

        # 誤差拡散法
        for h in self.hidden_neurons:
            h.update_weight(diff, direct)
        self.out_neuron.update_weight(diff, direct)

        return diff
    

def main_xor():
    model = ThreeLayerModel(2, hidden_num=6)

    # --- train loop
    dataset = [
        [0, 0, 1.0],
        [1, 0, 0.0],
        [0, 1, 0.0],
        [1, 1, 1.0],
    ]
    total_error = 0
    display_interval = 1  # 誤差を表示する間隔
    error_history = []  # エラー率の履歴を保存するリスト

    iteration = 200

    for i in range(iteration):
        x1, x2, target = dataset[random.randint(0, len(dataset)) - 1]
        metric = model.train([x1, x2], target)
        total_error += metric

        # --- predict
        y = model.forward([x1, x2])
        print(f"{i} in[{x1:5.2f},{x2:5.2f}] -> {y:5.2f}, target {target:5.2f}, metric {metric:5.2f}")

        # 学習回数に応じて誤差を表示
        if (i + 1) % display_interval == 0:
            average_error = total_error / display_interval
            print(f"Average error after {i + 1} iterations: {average_error:.5f}")
            error_history.append(average_error)
            total_error = 0  # 誤差をリセット

    # エラー率の遷移をグラフで表示
    plt.plot(range(display_interval, iteration + 1, display_interval), error_history, marker='o')
    plt.title('Error Rate Transition')
    plt.xlabel('Iteration')
    plt.ylabel('Average Error')
    plt.grid(True)
    plt.show()

    print("--- result ---")
    for x1, x2, target in dataset:
        y = model.forward([x1, x2])
        print(f"[{x1:5.2f},{x2:5.2f}] -> {y:5.2f}, target {target:5.2f}")

    # --- last weights
    print("--- output weights ---")
    print(model.out_neuron)

    print("--- hidden weights ---")
    for n in model.hidden_neurons:
        print(n)

if __name__ == "__main__":
    main_xor()