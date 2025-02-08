import math
import matplotlib.pyplot as plt
import japanize_matplotlib

def sigmoid(x, u0=0.4):
    # シグモイド関数の定義： s(x) = 1 / (1 + exp(-2*x/u0))
    return 1 / (1 + math.exp(-2 * x / u0))

def main():

  # パラメーターの初期化
  alpha = 0.5
  bias = 0.8
  hidden_weights = [-0.25, 0.42, -0.2, 0.38, -0.25, 0.34]
  output_weights = [0.74, -0.37, -0.59]

  inputs_value = [1, 1]

  
  # 
  # forward
  #  

  # 入力の設定
  inputs1_p = inputs_value[0]
  inputs1_n = inputs_value[0]

  inputs2_p = inputs_value[1]
  inputs2_n = inputs_value[1]

  # バイアスの設定
  hidden_bias_p = bias
  hidden_bias_n = bias

  # 初期のフォワードパス
  hidden_n = (inputs1_p * hidden_weights[2] + inputs1_n * hidden_weights[3] +
              inputs2_p * hidden_weights[4] + inputs2_n * hidden_weights[5] +
              hidden_bias_p * hidden_weights[0] + hidden_bias_n * hidden_weights[1])
  hidden_n = sigmoid(hidden_n)
  print(f"初期中間層出力: {hidden_n}")

  output_p = hidden_n * output_weights[2] + hidden_bias_p * output_weights[0] + hidden_bias_n * output_weights[1]
  output_p = sigmoid(output_p)
  print(f"初期出力: {output_p}")

  error = 1 - output_p
  print(f"初期誤差: {error}")

  dataset = [
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ]

  # トレーニングの設定（エポック数など）
  epochs = 100  # エポック数を適宜変更してください
  error_list = []

  for i in range(epochs):
      for x1, x2, target in dataset:
        print(f"--- エポック {i} ---")
        
        # フォワードパス
        hidden_n = (x1 * hidden_weights[2] + x1 * hidden_weights[3] +
                    x2 * hidden_weights[4] + x2 * hidden_weights[5] +
                    hidden_bias_p * hidden_weights[0] + hidden_bias_n * hidden_weights[1])
        hidden_n = sigmoid(hidden_n)

        output_p = hidden_n * output_weights[2] + hidden_bias_p * output_weights[0] + hidden_bias_n * output_weights[1]
        output_p = sigmoid(output_p)
        
        # 誤差の計算（例としてターゲットを 1 としています）
        error = target - output_p
        error_list.append(error)
        print(f"誤差: {error}")

        # トレーニング処理（重みの更新ロジックは実装例として出力のみ）
        hidden_upper_idx_list = [0, 2, 4]
        hidden_lower_idx_list = [1, 3, 5]

        output_upper_idx_list = [0]
        output_lower_idx_list = [1, 2]

        direct = "" 
        if error > 0:
            direct = "upper"
        else:
            direct = "lower"

        if direct == "upper":
            for j in hidden_upper_idx_list:
                hidden_weights[j] += error * hidden_weights[j] * alpha
        else:
            for j in hidden_lower_idx_list:
                hidden_weights[j] -= error * hidden_weights[j] * alpha

        if direct == "upper":
            for j in output_upper_idx_list:
                output_weights[j] += error * output_weights[j] * alpha
        else:
            for j in output_lower_idx_list:
                output_weights[j] -= error * output_weights[j] * alpha

        print(f"重み: {hidden_weights}")
        

      # for j in hidden_lower_idx_list:
      #   print(f"重み index {j}: {hidden_weights[j]}")
      #   next_weight = error
      #   print(f"次の重み (出力層): {next_weight}")




          

  # エラー推移のグラフ表示
  plt.plot(range(len(error_list)), error_list, marker='o')
  plt.xlabel("データポイント")
  plt.ylabel("誤差")
  plt.title("トレーニング中の誤差推移")
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  main()


