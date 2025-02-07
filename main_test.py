# main.pyのコードの実験用
# 重みの初期値を固定して実験しやすくした
# XORはパラメーター数が違うため削除

# sigmoidの計算は元コードより、一般的な形と少し違いました。-2/u0 って何でしょうかね?
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用

def sigmoid(x, u0=0.4):
    return 1 / (1 + math.exp(-2 * x / u0))

print("出力", sigmoid(0.406))
print("出力", sigmoid(-0.225507535979293))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

print("sigmoid微分した値", sigmoid_derivative(0.406))
print("sigmoid微分した値", sigmoid_derivative(-0.225507535979293))

class Neuron:
    def __init__(
        self,
        in_neurons: list["Neuron"],  # 入力ニューロンのリスト
        ntype: str,                  # ニューロンのタイプ: "p"はポジティブ、"n"はネガティブ
        round_weight,
        alpha: float = 0.8,          # 学習率 (多分)
        activation=sigmoid,          # 活性化関数
        activation_derivative=sigmoid_derivative,  # 活性化関数の微分
    ) -> None:
        self.ntype = ntype
        self.alpha = alpha
        self.activation = activation
        self.activation_derivative = activation_derivative

        # --- init weights
        # 0～1の乱数で初期化                                                                                                                          
        # 接続の種類で符号を変える: pp+ pn- np- nn+
        self.weights = []
        if round_weight:
            self.weights = round_weight
        else:
            for n in in_neurons:
                if ntype == "p":
                    if n.ntype == "p":
                        ope = 1
                    else:
                        ope = -1
                else:
                    if n.ntype == "p":
                        ope = -1
                    else:
                        ope = 1
                self.weights.append(round(random.random() * ope, 2))

        print("self.weights", self.weights)
        # --- operator
        # 元のowに相当、ここの符号はよくわからなかったので元コードをそのまま再現
        self.operator = 1 if ntype == "p" else -1
        self.weights_operator = [n.operator for n in in_neurons]

        # --- update index
        # 入力元が+ならupper時に学習
        # 入力元が-ならlower時に学習
        self.upper_idx_list = [] # 入力が+の場合のインデックス
        self.lower_idx_list = [] # 入力が-の場合のインデックス
        for i, n in enumerate(in_neurons):
            if n.ntype == "p":
                self.upper_idx_list.append(i)
            else:
                self.lower_idx_list.append(i)

    def forward(self, x):
        # 順方向の計算は既存と同じ
        assert len(self.weights) == len(x)
        y = [x[i] * self.weights[i] for i in range(len(self.weights))]
        y = sum(y)
        self.prev_in = x   # update用に一時保存
        self.prev_out = y  # update用に一時保存
        print("sigmoidなし", y)
        y = self.activation(y)
        return y

    def update_weight(self, delta_out, direct: str):
        # 誤差拡散による学習、逆伝搬というと怒られそう（笑）

        # f'(o)
        # 元コードではsigmoidを通した後の値を保存して利用することで少し軽量化している
        # 絶対値をとらなくても動くのでabs関数を外した
        grad = self.activation_derivative(self.prev_out)

        if direct == "upper":
            indices = self.upper_idx_list
        else:
            indices = self.lower_idx_list

        print("向き", direct)
        for idx in indices:
            # prev_inには前回の入力値が入っている
            # idxには1,3のようなupperかlowerのインデックスが入っている
            print("prev_in", self.prev_in)
            print("prev_out", self.prev_out)
            delta = self.alpha * self.prev_in[idx] 
            delta *= grad
            delta *= delta_out * self.operator * self.weights_operator[idx]
            self.weights[idx] += delta

    def __str__(self):
        return f"Neuron(type={self.ntype}, weights={self.weights})"

class ThreeLayerModel:
    def __init__(
        self,
        input_num: int,
        hidden_num: int,
        alpha: float = 0.8, # 学習率
        beta: float = 0.8, # bias
    ) -> None:
        self.beta = beta

        # 元コード上は [hd+, hd-] とprintされるもの
        # 多分bias?
        self.hd_p = Neuron([], "p", round_weight=[])
        self.hd_n = Neuron([], "n", round_weight=[])

        # input
        # 入力はpとnそれぞれを作成
        self.inputs: list[Neuron] = []  # inputsを属性として保存
        for i in range(input_num):
            self.inputs.append(Neuron([], "p", round_weight=[]))
            self.inputs.append(Neuron([], "n", round_weight=[]))

        # hidden
        # 入力は、[hd+, hd-, in1+, in1-, in2+, in2-, ...]
        self.hidden_neurons: list[Neuron] = []
        for i in range(hidden_num):
            self.hidden_neurons.append(
                Neuron(
                    [self.hd_p, self.hd_n] + self.inputs,
                    ntype=("p" if i % 2 == 1 else "n"),  # 元コードに合わせて-から作成
                    alpha=alpha,
                    round_weight=[-0.25, 0.42, -0.2, 0.38, -0.25, 0.34],
                )
            )

        # output
        # 入力は [hd+, hd-, h1-, h2+, h3-, ...]
        self.out_neuron = Neuron([self.hd_p, self.hd_n] + self.hidden_neurons, "p", alpha=alpha, round_weight=[0.74, -0.37, -0.59])

    def forward(self, inputs):
        # 入力用の配列を作成、入力をp用とn用に複製
        x = []
        for n in inputs:
            x.append(n)  # p
            x.append(n)  # n

        # hidden layerのforward
        # 入力に [hd+, hd-] も追加
        x = [h.forward([self.beta, self.beta] + x) for h in self.hidden_neurons]
        print("中間層のforward", x)
        # out layer forward
        # 入力に [hd+, hd-] も追加
        x = self.out_neuron.forward([self.beta, self.beta] + x)
        print("出力層のforward", x)
        return x

    def train(self, inputs, target):
        x = self.forward(inputs)

        # --- update(ED)
        # 差分を取得し、更新方向を見る
        diff = target - x
        if diff > 0:
            direct = "upper"
        else:
            direct = "lower"
            diff = -diff
        print("diff", diff)
        # 各ニューロンを更新
        # 中間層から更新しても良いし、出力層から更新しても良い

        # 中間層
        for h in self.hidden_neurons:
            h.update_weight(diff, direct)

        # 出力層
        self.out_neuron.update_weight(diff, direct)

        return diff
    

def plot_decision_boundary(model: ThreeLayerModel, dataset: list[list[float]]):
    # 決定境界をプロットするための関数
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01  # ステップサイズ

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.array([model.forward([x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-0.1, 0.5, 1.1], colors=['#FFAAAA', '#AAAAFF'], alpha=0.8)
    plt.scatter([x[0] for x in dataset], [x[1] for x in dataset], c=[x[2] for x in dataset], edgecolors='k', marker='o')
    plt.title('Decision Boundary')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.grid(True)
    plt.show()

    # 3Dで決定境界をプロットするための関数
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.1  # ステップサイズ

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.array([model.forward([x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.8)
    ax.scatter([x[0] for x in dataset], [x[1] for x in dataset], [x[2] for x in dataset], c='r', marker='o')
    ax.set_title('3D Decision Boundary')
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Output')
    plt.show()

def main_and():
    model = ThreeLayerModel(2, hidden_num=1)

    # --- train loop
    dataset = [
        [1, 1, 1.0],
        [0, 0, 0.0],
        [1, 0, 0.0],
        [0, 1, 0.0],
    ]
    total_error = 0
    display_interval = 1  # 誤差を表示する間隔
    error_history = []  # エラー率の履歴を保存するリスト

    # iteration = 200

    # for i in range(iteration):
    #     x1, x2, target = dataset[random.randint(0, len(dataset)) - 1]
    #     metric = model.train([x1, x2], target)
    #     total_error += metric

    #     # --- predict
    #     y = model.forward([x1, x2])
    #     print(f"{i} in[{x1:5.2f},{x2:5.2f}] -> {y:5.2f}, target {target:5.2f}, metric {metric:5.2f}")

    #     # 学習回数に応じて誤差を表示
    #     if (i + 1) % display_interval == 0:
    #         average_error = total_error / display_interval
    #         print(f"Average error after {i + 1} iterations: {average_error:.5f}")
    #         error_history.append(average_error)
    #         total_error = 0  # 誤差をリセット


    iteration = 1
    # ランダム選択を削除し、順番にループする

    for i in range(iteration):
        for x1, x2, target in dataset:
            metric = model.train([x1, x2], target)
            total_error += metric

            # --- predict
            y = model.forward([x1, x2])
            print(f"{i} in[{x1:5.2f},{x2:5.2f}] -> {y:5.2f}, target {target:5.2f}, metric {metric:5.2f}")

        # 学習回数に応じて誤差を表示
        if (i + 1) % display_interval == 0:
            average_error = total_error / (display_interval * len(dataset))  # データセットのサイズで割る
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

    # 決定境界をプロット
    plot_decision_boundary(model, dataset)

    # --- input weights
    print("--- hd weights ---")
    print(model.hd_p)
    print(model.hd_n)

    # --- input weights
    print("--- input weights ---")
    for n in model.inputs:
        print(n)

    print("--- hidden weights ---")
    for n in model.hidden_neurons:
        print(n)
    
    # --- last weights
    print("--- output weights ---")
    print(model.out_neuron)

def main_or():
    model = ThreeLayerModel(2, hidden_num=1)

    # --- train loop
    dataset = [
        [0, 0, 0.0],
        [1, 0, 1.0],
        [0, 1, 1.0],
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


    # --- input weights
    print("--- hd weights ---")
    print(model.hd_p)
    print(model.hd_n)

    # --- input weights
    print("--- input weights ---")
    for n in model.inputs:
        print(n)

    print("--- hidden weights ---")
    for n in model.hidden_neurons:
        print(n)
    
    # --- last weights
    print("--- output weights ---")
    print(model.out_neuron)

def main_not():
    model = ThreeLayerModel(2, hidden_num=1)

    # --- train loop
    dataset = [
        [0, 0, 1.0],
        [1, 1, 0.0],
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

    # --- input weights
    print("--- hd weights ---")
    print(model.hd_p)
    print(model.hd_n)

    # --- input weights
    print("--- input weights ---")
    for n in model.inputs:
        print(n)

    print("--- hidden weights ---")
    for n in model.hidden_neurons:
        print(n)
    
    # --- last weights
    print("--- output weights ---")
    print(model.out_neuron)

if __name__ == "__main__":
    main_and()