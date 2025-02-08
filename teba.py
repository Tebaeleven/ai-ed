import math

def sigmoid(x, u0=0.4):
    # シグモイド関数の定義： s(x) = 1 / (1 + exp(-2*x/u0))
    return 1 / (1 + math.exp(-2 * x / u0))

def main():

  # パラメーターの初期化
  bias = 0.8
  hidden_weights = [-0.25, 0.42, -0.2, 0.38, -0.25, 0.34]
  output_weights = [0.74, -0.37, -0.59]

  inputs_value = [1, 1]

  # forward

  # 入力の設定
  inputs1_p = inputs_value[0]
  inputs1_n = inputs_value[0]

  inputs2_p = inputs_value[1]
  inputs2_n = inputs_value[1]

  # バイアスの設定
  hidden_bias_p = bias
  hidden_bias_n = bias

  # 中間層の計算
  hidden_n = inputs1_p * hidden_weights[2] + inputs1_n * hidden_weights[3] + inputs2_p * hidden_weights[4] + inputs2_n * hidden_weights[5] + hidden_bias_p * hidden_weights[0] + hidden_bias_n * hidden_weights[1]
  hidden_n = sigmoid(hidden_n)
  print(f"中間層: {hidden_n}")
  
  # 出力層の計算
  output_p = hidden_n * output_weights[2] + hidden_bias_p * output_weights[0] + hidden_bias_n * output_weights[1]
  output_p = sigmoid(output_p)
  print(f"出力: {output_p}")

  error = 1 - output_p
  print(f"誤差: {error}")

if __name__ == "__main__":
  main()


