# sigmoidの計算は元コードより、一般的な形と少し違いました。-2/u0 って何でしょうかね?
import math
import random
import matplotlib.pyplot as plt

def sigmoid(x, u0=0.4):
    return 1 / (1 + math.exp(-2 * x / u0))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    def __init__(
        self,
        in_neurons: list["Neuron"],  # 入力ニューロンのリスト
        ntype: str,                  # ニューロンのタイプ: "p"はポジティブ、"n"はネガティブ
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
            # self.weights.append(0.2 * ope)
            self.weights.append(random.random() * ope)

        # --- operator
        # 元のowに相当、ここの符号はよくわからなかったので元コードをそのまま再現
        self.operator = 1 if ntype == "p" else -1
        self.weights_operator = [n.operator for n in in_neurons]

        # --- update index
        # 入力元が+ならupper時に学習
        # 入力元が-ならlower時に学習
        self.upper_idx_list = []
        self.lower_idx_list = []
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
        y = self.activation(y)
        return y

    def update_weight(self, delta_out, direct: str):
        # 誤差拡散による学習、逆伝搬というと怒られそう（笑）

        # f'(o)
        # 元コードではsigmoidを通した後の値を保存して利用することで少し軽量化している
        grad = self.activation_derivative(abs(self.prev_out))

        if direct == "upper":
            indices = self.upper_idx_list
        else:
            indices = self.lower_idx_list

        for idx in indices:
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
        alpha: float = 0.8,
        beta: float = 0.8,
    ) -> None:
        self.beta = beta

        # 元コード上は [hd+, hd-] とprintされるもの
        # 多分bias?
        self.hd_p = Neuron([], "p")  # hd_pを属性として保存
        self.hd_n = Neuron([], "n")  # hd_nを属性として保存

        # input
        # 入力はpとnそれぞれを作成
        self.inputs: list[Neuron] = []  # inputsを属性として保存
        for i in range(input_num):
            self.inputs.append(Neuron([], "p"))
            self.inputs.append(Neuron([], "n"))

        # hidden
        # 入力は、[hd+, hd-, in1+, in1-, in2+, in2-, ...]
        self.hidden_neurons: list[Neuron] = []
        for i in range(hidden_num):
            self.hidden_neurons.append(
                Neuron(
                    [self.hd_p, self.hd_n] + self.inputs,
                    ntype=("p" if i % 2 == 1 else "n"),  # 元コードに合わせて-から作成
                    alpha=alpha,
                )
            )

        # output
        # 入力は [hd+, hd-, h1-, h2+, h3-, ...]
        self.out_neuron = Neuron([self.hd_p, self.hd_n] + self.hidden_neurons, "p", alpha=alpha)

    def forward(self, inputs):
        # 入力用の配列を作成、入力をp用とn用に複製
        x = []
        for n in inputs:
            x.append(n)  # p
            x.append(n)  # n

        # hidden layerのforward
        # 入力に [hd+, hd-] も追加
        x = [h.forward([self.beta, self.beta] + x) for h in self.hidden_neurons]

        # out layer forward
        # 入力に [hd+, hd-] も追加
        x = self.out_neuron.forward([self.beta, self.beta] + x)

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
        
        # 各ニューロンを更新
        for h in self.hidden_neurons:
            h.update_weight(diff, direct)
        self.out_neuron.update_weight(diff, direct)

        return diff
    

def main_xor():
    model = ThreeLayerModel(2, hidden_num=4)

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

    # --- input weights
    print("--- hd weights ---")
    print(model.hd_p)
    print(model.hd_n)

    # --- last weights
    print("--- output weights ---")
    print(model.out_neuron)

    print("--- hidden weights ---")
    for n in model.hidden_neurons:
        print(n)

def main_and():
    model = ThreeLayerModel(2, hidden_num=1)

    # --- train loop
    dataset = [
        [0, 0, 0.0],
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