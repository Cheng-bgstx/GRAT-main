import matplotlib
# 【新增】强制使用 Agg 后端，这个后端专门用于生成文件，不依赖图形窗口
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体 (尝试使用系统常见字体，如果乱码请根据环境调整)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def draw_manifold_concept():
    fig = plt.figure(figsize=(15, 6), dpi=150)

    # --- 子图 1: 离线锚点 (The Anchor) ---
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Offline Phase:\nUniversal Anchor (Static Point)", fontsize=12, fontweight='bold')

    # 模拟一个通用噪声图
    noise = np.random.randn(20, 20)
    # 加上一些结构性纹理模拟医学通用弱点
    for i in range(20):
        noise[i, i] += 2.0
        noise[i, 19 - i] += 2.0

    im1 = ax1.imshow(noise, cmap='coolwarm')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(10, 22, r"$\delta_{anchor}$ (Pixel Space)", ha='center', fontsize=11)
    ax1.text(10, 24, "Absolute Position\nFixed Texture", ha='center', fontsize=10, color='gray')

    # 添加一个箭头指向子图2
    trans = fig.transFigure
    arrow = patches.FancyArrowPatch((0.32, 0.5), (0.36, 0.5), transform=trans,
                                    connectionstyle="arc3,rad=0",
                                    arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black")
    fig.patches.append(arrow)

    # --- 子图 2: 转化过程 (Transformation) ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Transformation:\nPoint to Direction", fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 画向量归一化示意图
    ax2.text(0.5, 0.8, r"$m_0 = \frac{\delta_{anchor}}{\| \delta_{anchor} \|_1 + \epsilon}$", ha='center', fontsize=16)

    # 画一个大箭头变小箭头的过程
    ax2.arrow(0.2, 0.4, 0.4, 0.2, head_width=0.05, head_length=0.05, fc='lightgray', ec='gray',
              label='Original Magnitude')
    ax2.arrow(0.2, 0.4, 0.2, 0.1, head_width=0.05, head_length=0.05, fc='blue', ec='blue', width=0.015,
              label='Normalized Momentum')

    ax2.text(0.5, 0.2, "Extracting Pure Direction\n(Unit Vector)", ha='center', fontsize=11, color='blue')
    ax2.text(0.5, 0.1, "Discard Magnitude,\nKeep Structure", ha='center', fontsize=10, color='gray')

    # 添加一个箭头指向子图3
    arrow2 = patches.FancyArrowPatch((0.64, 0.5), (0.68, 0.5), transform=trans,
                                     connectionstyle="arc3,rad=0",
                                     arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8", color="black")
    fig.patches.append(arrow2)

    # --- 子图 3: 在线攻击流形 (Online Manifold) ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Online Phase:\nInstance Adaptation", fontsize=12, fontweight='bold')

    # 创建损失地形图 (Loss Landscape)
    x = np.linspace(-2, 3, 100)
    y = np.linspace(-2, 3, 100)
    X, Y = np.meshgrid(x, y)
    # 模拟一个复杂的损失函数，最优解在 (2, 2)
    Z = (X - 2) ** 2 + (Y - 2) ** 2 + 0.5 * np.sin(3 * X)

    ax3.contourf(X, Y, Z, 20, cmap='viridis_r', alpha=0.6)

    # 起点 (Original Image)
    ax3.plot(0, 0, 'ko', markersize=8, label='Start (x)')

    # 1. 传统方法 (Additive / Warm Start)
    # 假设 Anchor 指向 (1.5, 0.5)，直接跳过去
    ax3.plot(1.5, 0.5, 'rx', markersize=8, markeredgewidth=2, label='Additive Start')
    ax3.annotate('Misaligned!', xy=(1.5, 0.5), xytext=(0.5, -1),
                 arrowprops=dict(facecolor='red', shrink=0.05), color='red')

    # 2. 你的方法 (Momentum / Manifold Init)
    # 初始动量指向通用方向 (蓝色箭头)
    ax3.arrow(0, 0, 0.8, 0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue', width=0.02,
              label='Initial Momentum ($m_0$)')

    # 模拟攻击路径 (绿色曲线)，受初始动量影响，顺滑地流向最优解
    path_x = np.linspace(0, 2, 20)
    path_y = 0.5 * path_x + 0.2 * path_x ** 2  # 模拟受动量惯性影响的弯曲路径
    ax3.plot(path_x, path_y, 'g-', linewidth=2, label='Guided Attack Path')

    ax3.plot(2, 2, 'w*', markersize=15, markeredgecolor='k', label='Target Weakness')

    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_xlim(-1, 3)
    ax3.set_ylim(-1.5, 3)
    ax3.text(1.5, 1.0, "Surfing the Manifold", color='green', rotation=30, fontweight='bold')

    plt.tight_layout()
    plt.savefig('manifold_initialization.png')
    # plt.show() # 如果在本地环境可以取消注释显示


draw_manifold_concept()