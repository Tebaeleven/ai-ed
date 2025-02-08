import math
import matplotlib.pyplot as plt
import japanize_matplotlib

def sigmoid(x, u0=0.4):
    # シグモイド関数の定義： s(x) = 1 / (1 + exp(-2*x/u0))
    return 1 / (1 + math.exp(-2 * x / u0))

def main():

  # パラメーターの初期化
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

  # トレーニングの設定（エポック数など）
  epochs = 1  # エポック数を適宜変更してください
  error_list = []

  for i in range(epochs):
      print(f"--- エポック {i} ---")
      
      # フォワードパス
      hidden_n = (inputs1_p * hidden_weights[2] + inputs1_n * hidden_weights[3] +
                  inputs2_p * hidden_weights[4] + inputs2_n * hidden_weights[5] +
                  hidden_bias_p * hidden_weights[0] + hidden_bias_n * hidden_weights[1])
      hidden_n = sigmoid(hidden_n)

      output_p = hidden_n * output_weights[2] + hidden_bias_p * output_weights[0] + hidden_bias_n * output_weights[1]
      output_p = sigmoid(output_p)
      
      # 誤差の計算（例としてターゲットを 1 としています）
      error = 1 - output_p
      error_list.append(error)
      print(f"誤差: {error}")

      # トレーニング処理（重みの更新ロジックは実装例として出力のみ）
      hidden_upper_idx_list = [0, 2, 4]
      hidden_lower_idx_list = [1, 3, 5]
      for j in hidden_upper_idx_list:
          print(f"重み index {j}: {hidden_weights[j]}")
          next_weight = output_weights[2]
          print(f"次の重み (出力層): {next_weight}")
          # ※ ※ ここに学習アルゴリズムに基づく重み更新ロジックを追加する ※ ※

  # エラー推移のグラフ表示
  plt.plot(range(epochs), error_list, marker='o')
  plt.xlabel("エポック")
  plt.ylabel("誤差")
  plt.title("トレーニング中の誤差推移")
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  main()


