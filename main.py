import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定数
c = 3e8  # 光速 [m/s]
frequency = 27.9e9  # 周波数 [Hz]
wavelength = c / frequency  # 波長 [m]
dx = dy = dz = wavelength / 20  # 空間刻み [m]
dt = dx / (2 * c)  # 時間刻み [s]

# 計算領域のサイズ（メートル）
size_x = 0.5  # 計算領域を1mから0.5mに縮小
size_y = 0.5
size_z = 0.5

# グリッドのサイズ
Nx = int(size_x / dx)
Ny = int(size_y / dy)
Nz = int(size_z / dz)
max_time = 100  # シミュレーションの時間ステップ数

# 比誘電率と比透磁率
epsilon_air = 1.0
mu_air = 1.0
epsilon_antenna = 1.0
mu_antenna = 1.0
epsilon_wall = 4.0  # 例として壁の比誘電率を設定
mu_wall = 1.0

# 配列の初期化（データ型をfloat32に変更）
Ex = np.zeros((Nx, Ny, Nz), dtype=np.float32)
Ey = np.zeros((Nx, Ny, Nz), dtype=np.float32)
Ez = np.zeros((Nx, Ny, Nz), dtype=np.float32)
Hx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
Hy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
Hz = np.zeros((Nx, Ny, Nz), dtype=np.float32)

# 壁の設定
wall_start_x = int(0.8 * Nx)
wall_end_x = wall_start_x + int(0.5 * Nx / size_x)
wall_start_y = int(0.25 * Ny)
wall_end_y = wall_start_y + int(0.5 * Ny)
wall_start_z = int(0.25 * Nz)
wall_end_z = wall_start_z + int(0.5 * Nz)
wall = np.zeros((Nx, Ny, Nz), dtype=np.float32)
wall[wall_start_x:wall_end_x, wall_start_y:wall_end_y, wall_start_z:wall_end_z] = epsilon_wall

# アンテナの位置
antenna_x = Nx // 2
antenna_y = Ny // 2
antenna_z = int(0.5 * Nz)

# シミュレーション実行
for t in tqdm(range(max_time), desc="Simulating"):
    # 磁場の更新
    Hx[:, :, :-1] += (Ez[:, :, 1:] - Ez[:, :, :-1]) * dt / (mu_air * dz)
    Hy[:, :-1, :] += (Ex[:, 1:, :] - Ex[:, :-1, :]) * dt / (mu_air * dx)
    Hz[:-1, :, :] += (Ey[1:, :, :] - Ey[:-1, :, :]) * dt / (mu_air * dy)
    
    # 電場の更新
    Ex[:, 1:-1, 1:-1] += (Hz[:, 1:-1, 1:-1] - Hz[:, :-2, 1:-1]) * dt / (epsilon_air * dy)
    Ey[1:-1, :, 1:-1] += (Hx[1:-1, :, 1:-1] - Hx[1:-1, :, :-2]) * dt / (epsilon_air * dz)
    Ez[1:-1, 1:-1, :] += (Hy[1:-1, 1:-1, :] - Hy[:-2, 1:-1, :]) * dt / (epsilon_air * dx)
    
    # 壁の反射
    Ex[wall_start_x:wall_end_x, wall_start_y:wall_end_y, wall_start_z:wall_end_z] *= epsilon_wall / epsilon_air
    Ey[wall_start_x:wall_end_x, wall_start_y:wall_end_y, wall_start_z:wall_end_z] *= epsilon_wall / epsilon_air
    Ez[wall_start_x:wall_end_x, wall_start_y:wall_end_y, wall_start_z:wall_end_z] *= epsilon_wall / epsilon_air
    
    # アンテナのソース
    Ez[antenna_x, antenna_y, antenna_z] += 10 * np.sin(2 * np.pi * frequency * t * dt)  # 振幅を増やす

# 電場と磁場のプロット
def plot_fields(field, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = np.min(field), np.max(field)
    ax[0].imshow(field[:, :, Nz // 2], cmap='RdBu', vmin=vmin, vmax=vmax)
    ax[0].set_title(f'{title} (x-y plane)')
    ax[1].imshow(field[:, Ny // 2, :], cmap='RdBu', vmin=vmin, vmax=vmax)
    ax[1].set_title(f'{title} (x-z plane)')
    ax[2].imshow(field[Nx // 2, :, :], cmap='RdBu', vmin=vmin, vmax=vmax)
    ax[2].set_title(f'{title} (y-z plane)')
    plt.show()

plot_fields(Ez, 'Electric Field Ez')
plot_fields(Hx, 'Magnetic Field Hx')
plot_fields(Hy, 'Magnetic Field Hy')
plot_fields(Hz, 'Magnetic Field Hz')
