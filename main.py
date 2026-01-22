# simulate.py
# Моделирование движения материальной точки в центральном гравитационном поле
# + линейное сопротивление среды и (опционально) тяга двигателя вдоль скорости.
#
# Уравнения:
#   r = sqrt(x^2 + y^2)
#   dx/dt = vx
#   dy/dt = vy
#   dvx/dt = -mu * x / r^3 + (k_thrust(t,r) - k_drag) * vx
#   dvy/dt = -mu * y / r^3 + (k_thrust(t,r) - k_drag) * vy
#
# Численный метод: Рунге–Кутта 4-го порядка (RK4).
#
# Выход: results/trajectory.png, results/animation.gif (если включено сохранение).

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class Params:
    mu: float                  # GM (в "условных" единицах)
    k_drag: float              # коэффициент сопротивления (>=0)
    k_thrust0: float           # тяга до переключения
    k_thrust1: float           # тяга после переключения
    r_switch: float            # радиус переключения тяги (<=0 отключает)
    t_switch: float            # время переключения тяги (<=0 отключает)
    dt: float                  # шаг интегрирования
    t_max: float               # максимальное время
    r_stop: float              # остановка, если r < r_stop (падение на центр)
    max_steps: int             # ограничение по шагам


def k_thrust(t: float, r: float, p: Params) -> float:
    use_r = p.r_switch > 0.0
    use_t = p.t_switch > 0.0

    if not use_r and not use_t:
        return p.k_thrust0

    switched = False
    if use_r and r <= p.r_switch:
        switched = True
    if use_t and t >= p.t_switch:
        switched = True

    return p.k_thrust1 if switched else p.k_thrust0


def deriv(state: np.ndarray, t: float, p: Params) -> np.ndarray:
    """Правая часть ОДУ. state = [x, y, vx, vy]."""
    x, y, vx, vy = state
    r2 = x*x + y*y
    r = math.sqrt(r2)

    if r < 1e-12:
        return np.array([vx, vy, 0.0, 0.0], dtype=float)

    ax_g = -p.mu * x / (r*r*r)
    ay_g = -p.mu * y / (r*r*r)

    kt = k_thrust(t, r, p)
    kv = (kt - p.k_drag)

    ax = ax_g + kv * vx
    ay = ay_g + kv * vy

    return np.array([vx, vy, ax, ay], dtype=float)


def rk4_step(state: np.ndarray, t: float, p: Params) -> np.ndarray:
    dt = p.dt
    k1 = deriv(state, t, p)
    k2 = deriv(state + 0.5*dt*k1, t + 0.5*dt, p)
    k3 = deriv(state + 0.5*dt*k2, t + 0.5*dt, p)
    k4 = deriv(state + dt*k3, t + dt, p)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def simulate(x0: float, y0: float, vx0: float, vy0: float, p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает массивы t, x, y."""
    state = np.array([x0, y0, vx0, vy0], dtype=float)

    ts: List[float] = []
    xs: List[float] = []
    ys: List[float] = []

    t = 0.0
    for _ in range(p.max_steps):
        x, y, vx, vy = state
        r = math.hypot(x, y)

        ts.append(t)
        xs.append(x)
        ys.append(y)

        if t >= p.t_max:
            break
        if r <= p.r_stop:
            break

        state = rk4_step(state, t, p)
        t += p.dt

    return np.asarray(ts), np.asarray(xs), np.asarray(ys)


def ensure_results_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_trajectory_png(xs: np.ndarray, ys: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.scatter([0], [0], s=30)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Траектория")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_animation_gif(xs: np.ndarray, ys: np.ndarray, out_path: str, fps: int = 30, tail: int = 300) -> None:
    fig, ax = plt.subplots()

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    span = max(x_max - x_min, y_max - y_min)
    if span <= 1e-12:
        span = 1.0
    pad = 0.1 * span

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Движение точки")

    ax.scatter([0], [0], s=40)

    (line,) = ax.plot([], [], lw=1.5)
    (point,) = ax.plot([], [], marker="o", markersize=6)

    n = len(xs)

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame: int):
        i0 = max(0, frame - tail)
        line.set_data(xs[i0:frame+1], ys[i0:frame+1])
        point.set_data([xs[frame]], [ys[frame]])
        return line, point

    anim = FuncAnimation(fig, update, frames=n, init_func=init, interval=1000/fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def interactive_inputs() -> Tuple[float, float, float, float]:
    def ask(name: str, default: float) -> float:
        s = input(f"{name} (по умолчанию {default}): ").strip()
        return default if s == "" else float(s)

    x0 = ask("x0", 1.0)
    y0 = ask("y0", 0.0)
    vx0 = ask("vx0", 0.0)
    vy0 = ask("vy0", 1.0)
    return x0, y0, vx0, vy0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Моделирование движения в поле тяготения (2D) + сопротивление/тяга.")
    ap.add_argument("--interactive", action="store_true", help="ввод начальных данных через input()")

    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--vx0", type=float, default=0.0)
    ap.add_argument("--vy0", type=float, default=1.0)

    ap.add_argument("--mu", type=float, default=1.0, help="GM (условные единицы)")
    ap.add_argument("--k_drag", type=float, default=0.0, help="коэфф. сопротивления среды")
    ap.add_argument("--k_thrust0", type=float, default=0.0, help="коэфф. тяги до переключения")
    ap.add_argument("--k_thrust1", type=float, default=0.0, help="коэфф. тяги после переключения")
    ap.add_argument("--r_switch", type=float, default=-1.0, help="радиус переключения (<=0 отключает)")
    ap.add_argument("--t_switch", type=float, default=-1.0, help="время переключения (<=0 отключает)")

    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--t_max", type=float, default=30.0)
    ap.add_argument("--r_stop", type=float, default=0.03, help="остановка при падении на центр")
    ap.add_argument("--max_steps", type=int, default=2_000_000)

    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--save_png", action="store_true", help="сохранить траекторию PNG")
    ap.add_argument("--save_gif", action="store_true", help="сохранить анимацию GIF")
    ap.add_argument("--show", action="store_true", help="показать график/анимацию (окно)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tail", type=int, default=300, help="длина хвоста в кадрах")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.interactive:
        x0, y0, vx0, vy0 = interactive_inputs()
    else:
        x0, y0, vx0, vy0 = args.x0, args.y0, args.vx0, args.vy0

    p = Params(
        mu=args.mu,
        k_drag=args.k_drag,
        k_thrust0=args.k_thrust0,
        k_thrust1=args.k_thrust1,
        r_switch=args.r_switch,
        t_switch=args.t_switch,
        dt=args.dt,
        t_max=args.t_max,
        r_stop=args.r_stop,
        max_steps=args.max_steps,
    )

    t, xs, ys = simulate(x0, y0, vx0, vy0, p)

    ensure_results_dir(args.results_dir)

    png_path = os.path.join(args.results_dir, "trajectory.png")
    gif_path = os.path.join(args.results_dir, "animation.gif")

    if args.save_png:
        save_trajectory_png(xs, ys, png_path)
        print(f"Saved: {png_path}")

    if args.save_gif:
        save_animation_gif(xs, ys, gif_path, fps=args.fps, tail=args.tail)
        print(f"Saved: {gif_path}")

    if args.show:
        plt.figure()
        plt.plot(xs, ys)
        plt.scatter([0], [0], s=30)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Траектория")
        plt.show()


if __name__ == "__main__":
    main()
