import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def plot_sigmoid():

    # シグモイド関数 (ベクトル演算用にnp.expを使用)
    def sigmoid_np(x, u0=0.4):
        return 1 / (1 + np.exp(-2 * x / u0))

    # シグモイド関数の微分 (正しい接線の傾きを得るために (2/u0) の係数を掛ける)
    def sigmoid_derivative_np(x, u0=0.4):
        s = sigmoid_np(x, u0)
        return (2 / u0) * s * (1 - s)

    # プロットするxの範囲（例：-2～2）
    x_range = np.linspace(-2, 2, 400)
    y_sigmoid = sigmoid_np(x_range)

    # 指定された点
    points = [0.406, -0.225507535979293]
    y_points = [sigmoid_np(x) for x in points]
    slopes = [sigmoid_derivative_np(x) for x in points]

    # プロット開始
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_sigmoid, label="Sigmoid関数", color="blue")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.title("Sigmoid関数と各点での接線")

    # 指定点を赤い点でプロット
    plt.scatter(points, y_points, color="red", zorder=5, label="指定点")

    # 各指定点における接線を描画
    for x0, y0, m in zip(points, y_points, slopes):
        # 接線を描くため、点の周辺 (x0±0.5) のx値を用意
        x_tangent = np.linspace(x0 - 0.5, x0 + 0.5, 100)
        y_tangent = y0 + m * (x_tangent - x0)
        plt.plot(x_tangent, y_tangent, linestyle="--", label=f"接線 (x={x0:.3f}, m={m:.3f})")
        # 接線の傾きの値をテキストで表示
        plt.text(x0, y0, f" m={m:.3f}", fontsize=10, color="green")

    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_sigmoid() 