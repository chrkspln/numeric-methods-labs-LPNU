import matplotlib.pyplot as plot
import numpy as np

C1: float = 52e-6
C2: float = 70e-6
L_min: float = 3
L_max: float = 30
i_min: int = 1
i_max: int = 2
R1: int = 17
R2: int = 41
R3: int = 45
R4: int = 85

current_time = 0
h = 1e-5

def U1(curr_time: float) -> float:
    a = 7e-3
    period = 2 * a
    phase = curr_time % period

    if 0 <= phase < a:
        # Rising slope (0 to 10)
        return 10 * (phase / a)
    else:
        return 0


def L2(i2: float) -> float:
    if abs(i2) <= i_min:
        return L_max
    elif i_min < abs(i2) < i_max:
        spline = cubic_spline(i2)
        return spline
    else:
        return L_min


differential_equations = [
    lambda t, i2, uc1, uc2: (((uc2 - 2 * uc1) / (2 * R2 + R4)) - i2 + (U1(t) - uc1) / R3) / C1 if C1 != 0 else 0,
    lambda t, i2, uc1, uc2: (U1(t) - uc1 - ((uc2 - 2 * uc1) / (2 * R2 + R4) * R2) - ((uc2 - 2 * uc1) / (2 * R2 + R4) * R4)) / max(L2(i2), 1e-6),
    lambda t, i2, uc1, uc2: (((uc2 - 2 * uc1) / (2 * R2 + R4)) * C2)
]

spline_coefficients: list = [
    lambda x, x1, x2, h: (2 * (x - x1) + h) * pow((x2 - x), 2),
    lambda x, x1, x2, h: (2 * (x2 - x) + h) * pow((x - x1), 2),
    lambda x, x1, x2, h: (x - x1) * pow((x2 - x), 2),
    lambda x, x1, x2, h: (x - x2) * pow((x - x1), 2)
]
cubic_spline_y = lambda spline_coefficients, y1, y2, h, m1, m2: ((spline_coefficients[0] * y1 +
                                                                 spline_coefficients[1] * y2) / pow(h, 3) +
                                                                 (spline_coefficients[2] * m1 +
                                                                  spline_coefficients[3] * m2) / pow(h, 2))

def cubic_spline(i2: float):
    x: float = i2
    x1: float = i_min
    x2: float = i_max
    h: float = x2 - x1
    m1: float = (L_max - L_min) / h
    m2: float = 0
    coefficients: list = [0 for _ in range(4)]

    for i in range(len(coefficients)):
        coefficients[i] = spline_coefficients[i](x, x1, x2, h)

    return L_max + L_min - cubic_spline_y(coefficients, L_min, L_max, h, m1, m2)


def get_runge_kutta_coefficients(previous: list[float]) -> list[list[float]]:
    length: int = len(previous)
    new_coefficients: list[list[float]] = [[0 for _ in range(length)] for _ in range(4)]
    # K1
    for i in range(length):
        new_coefficients[0][i] = h * differential_equations[i](current_time, previous[0], previous[1], previous[2])
    # K2
    for i in range(length):
        new_coefficients[1][i] = h * differential_equations[i](current_time + h / 2,
                                                               previous[0] + new_coefficients[0][0] / 2,
                                                               previous[1] + new_coefficients[0][1] / 2,
                                                               previous[2] + new_coefficients[0][2] / 2)
    # K3
    for i in range(length):
        new_coefficients[2][i] = h * differential_equations[i](current_time + h / 2,
                                                               previous[0] + new_coefficients[1][0] / 2,
                                                               previous[1] + new_coefficients[1][1] / 2,
                                                               previous[2] + new_coefficients[1][2] / 2)
    # K4
    for i in range(length):
        new_coefficients[3][i] = h * differential_equations[i](current_time + h,
                                                               previous[0] + new_coefficients[2][0],
                                                               previous[1] + new_coefficients[2][1],
                                                               previous[2] + new_coefficients[2][2])

    return new_coefficients


def runge_kutta_formula(current_args: list[float]) -> list[float]:
    global current_time
    coefficients: list[list[float]] = get_runge_kutta_coefficients(current_args)
    new_args: list[float] = []

    def calculate_through_coefficients(k):
        return (k[0] + 2 * k[1] + 2 * k[2] + k[3]) / 6

    for j in range(len(current_args)):
        k = []
        for i in range(len(coefficients)):
            k.append(coefficients[i][j])
        new_args.append(current_args[j] + calculate_through_coefficients(k))

    current_time += h
    return new_args


def build_graph(x_ax, y_ax, y_ax_name, x_ax_name='t (s)'):
    ax = plot.gca()
    plot.rcParams['font.family'] = 'serif'
    plot.rcParams['font.serif'] = ['Times New Roman']
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([x_ax[0], x_ax[-1]])
    plot.plot(x_ax, y_ax)
    plot.ylabel(y_ax_name)
    plot.xlabel(x_ax_name)
    plot.grid(True)
    plot.show()


def main():
    time = np.arange(0, 0.007, h)
    res_Uc1, res_i2, res_Uc2 = [], [], []

    current_values = [0, 0, 0]
    for t in time:
        res_Uc1.append(current_values[0])
        res_i2.append(current_values[1])
        res_Uc2.append(current_values[2])

        # Debug: Print current values each iteration
        print(
            f"Time: {t:.6f}s, Uc1: {current_values[0]:.4e}, i2: {current_values[1]:.4e}, Uc2: {current_values[2]:.4e}"
        )

        current_values = runge_kutta_formula(current_values)

    # Adjust time length to match result arrays for plotting
    time = time[:len(res_Uc1)]

    # Plot each graph with corrected x and y ranges
    build_graph(time, res_Uc1, "Uс1")
    build_graph(time, res_i2, "i2")
    build_graph(time, res_Uc2, "U2")
    build_graph([i * 0.001 for i in range(0, 3000)], [L2(i * 0.001) for i in range(0, 3000)], "L2", "i")

    print("U1 range:", min([U1(t) for t in [i * h for i in range(20000)]]),
          max([U1(t) for t in [i * h for i in range(20000)]]))
    print("Uc1 range:", min(res_Uc1), max(res_Uc1))
    print("i2 range:", min(res_i2), max(res_i2))
    print("U2 range:", min(res_Uc2), max(res_Uc2))
    print("L2 range:", min([L2(i * 0.001) for i in range(0, 3000)]), max([L2(i * 0.001) for i in range(0, 3000)]))


if __name__ == '__main__':
    main()
