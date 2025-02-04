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
        

        # もし誤差が0より大きいなら増やす、0より小さいなら減らす
        if error > 0:
            # 重みを増やす
            pass
        else:
            # 重みを減らす
            pass
          

    def relu(self, x):
      return max(0, x)
    
def main():
    
    # 入力
    x1 = 1.5
    x2 = 0

    # ニューロン1段目
    unit1 = NeuronUnit(name="unit1", x=x1)
    unit1.forward()
    print(unit1.y)

    unit1.train(unit1.y, 1)

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
