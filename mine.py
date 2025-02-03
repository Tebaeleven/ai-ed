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
    def __init__(self, input_neurons: list["Neuron"], output_neurons: list["Neuron"], type: Literal["p", "n"]) -> None:
        self.type = type
        self.activation = sigmoid
        # 相手から繋がってきているニューロン
        self.input_neurons = input_neurons

        # 相手に繋げているニューロン
        self.output_neurons = output_neurons

        self.input_weights = []

        # --- init weights
        # 0～1の乱数で初期化                                                                                                                          
        # 接続の種類で符号を変える: pp+ pn- np- nn+
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
            self.input_weights.append(0.2 * ope)

        # output_weightsの初期化を後に移動
        self.output_weights = []

    def set_output_weights(self):
        # output_weightsをoutput_neuronsに基づいて初期化
        self.output_weights = [0.2 for _ in self.output_neurons]

    def forward(self, x):
        return sigmoid(x)
    
    def update_weight(self, delta_out, direct: str):
        pass

    def get_sum_output_weights(self,type: Literal["upper", "lower"]):
        if type == "upper":
            return sum([self.input_weights[i] for i in self.output_neurons])
        else:
            return sum([self.input_weights[i] for i in self.input_neurons])


class ThreeLayerModel:
    def __init__(self, input_num: int, hidden_num: int) -> None:
        hd_p = Neuron([], [], "p")
        hd_n = Neuron([], [], "n")

        # input
        self.inputs: list[Neuron] = []
        for i in range(input_num):
            self.inputs.append(Neuron([], [], "p"))
            self.inputs.append(Neuron([], [], "n"))

        # hidden
        self.hidden_neurons: list[Neuron] = []
        for i in range(hidden_num):
            self.hidden_neurons.append(
                Neuron(
                    [hd_p, hd_n] + self.inputs,
                    [],
                    type=("p" if i % 2 == 1 else "n"),
                )
            )

        # output
        self.out_neuron = Neuron([hd_p, hd_n] + self.hidden_neurons, [], "p")

        # それぞれのニューロンに接続先の設定をする
        for neuron in self.inputs:
            neuron.output_neurons.append(self.hidden_neurons[0])
            neuron.output_neurons.append(self.hidden_neurons[1])

        for neuron in self.hidden_neurons:
            neuron.output_neurons.append(self.out_neuron)

        # output_weightsを設定
        for neuron in self.inputs + self.hidden_neurons + [self.out_neuron]:
            neuron.set_output_weights()

    def display_neuron_parameters(self):
        for neuron in self.inputs:
            print("inputs")

            print(f"Neuron Type: {neuron.type}")
            print(f"Input Weights: {neuron.input_weights}")
            print(f"Output Weights: {neuron.output_weights}")
            print(neuron.type)

            print("-" * 30)
    
        for neuron in self.hidden_neurons:
            print("hidden")

            print(f"Neuron Type: {neuron.type}")
            print(f"Input Weights: {neuron.input_weights}")
            print(f"Output Weights: {neuron.output_weights}")
            print(neuron.type)

            print("-" * 30)


        neuron = self.out_neuron
        print("output")
        print(f"Neuron Type: {neuron.type}")
        print(f"Input Weights: {neuron.input_weights}")
        print(f"Output Weights: {neuron.output_weights}")
        print("-" * 30)



def main():
    model = ThreeLayerModel(input_num=2, hidden_num=2)
    model.display_neuron_parameters()

if __name__ == "__main__":
    main()