

def get_schedule(t_T, t_0, n_sample, n_steplength, debug=0):
    if n_steplength > 1:
        if not n_sample > 1:
            raise RuntimeError('n_steplength has no effect if n_sample=1')

    t = t_T
    times = [t]
    while t >= 0:
        t = t - 1
        times.append(t)
        n_steplength_cur = min(n_steplength, t_T - t)

        for _ in range(n_sample - 1):

            for _ in range(n_steplength_cur):
                t = t + 1
                times.append(t)
            for _ in range(n_steplength_cur):
                t = t - 1
                times.append(t)

    _check_times(times, t_0, t_T)

    if debug == 2:
        for x in [list(range(0, 50)), list(range(-1, -50, -1))]:
            _plot_times(x=x, times=[times[i] for i in x])

    return times

def _plot_times(x, times):
    import matplotlib.pyplot as plt
    plt.plot(x, times)
    plt.show()


# 1.2.2 完整性检查函数
def _check_times(times, t_0, t_T):
    # 检查开头是否递减、结尾是否为 -1
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]

    # 所有相邻的时间步差值必须是 1（回跳也是 1步1步 的走）
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # 所有时间步都必须在 [t_0, t_T] 范围内
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)

# 1.2.1 回跳的条件判断的封装函数
def judge_bool(t, jump_list, jump_length, start_resampling):
    if t not in jump_list:
        return False
    if t > start_resampling - jump_length:
        return False
    if jump_list[t] < 0:
        return False
    return True

# 1.1.1 构造跳跃字典
def build_jumps(t_T, length, count):
    return {j: count - 1 for j in range(0, t_T - length, length)}


# ------------ 1. RePaint 的核心，jump back 跳跃时间表（已重构） ------------
def get_schedule_jump(
    t_T, n_sample, jump_length, jump_n_sample, jump2_length=1, jump2_n_sample=1,
    jump3_length=1, jump3_n_sample=1, start_resampling=100000000
):
    # 1.1 从 t_T 到 0，每隔 jump_length 步设置一个“回跳点”，重采样 jump_n_sample - 1 次
    jumps  = build_jumps(t_T, jump_length, jump_n_sample)
    jumps2 = build_jumps(t_T, jump2_length, jump2_n_sample)
    jumps3 = build_jumps(t_T, jump3_length, jump3_n_sample)

    # 1.2 循环跳跃，重构时间步
    t = t_T  # 以 t_T = 250 为例子
    ts = []  # 跳跃时间表
    while t >= 1:
        t = t - 1
        ts.append(t)

        # 普通 resample：每个时间点做 n_sample 次
        if (t + 1 < t_T - 1 and t <= start_resampling):
            for _ in range(n_sample - 1):
                ts.append(t + 1)
                ts.append(t)

        # jump3：更细粒度的回跳（扩展接口）
        if judge_bool(t, jumps3, jump3_length, start_resampling):
            jumps3[t] -= 1
            for i in range(1, jump3_length + 1):
                ts.append(t + i)

        # jump2：中级回跳 + 重置 jump3（扩展接口）
        if judge_bool(t, jumps2, jump2_length, start_resampling):
            jumps2[t] -= 1
            for i in range(1, jump2_length + 1):
                ts.append(t + i)
            jumps3 = build_jumps(jump3_length, jump3_n_sample)

        # jump：最粗粒度回跳 + 重置 jump2 & jump3（实现）
        if judge_bool(t, jumps, jump_length, start_resampling):
            jumps[t] -= 1
            for i in range(1, jump_length + 1):
                ts.append(t + i)
            jumps2 = build_jumps(jump2_length, jump2_n_sample)
            jumps3 = build_jumps(jump3_length, jump3_n_sample)

    ts.append(-1)  # 模型会在 -1 时结束循环
    _check_times(ts, -1, t_T)  # 断言检查函数，确保所有 ts 都合法
    return ts


def get_schedule_jump_paper():
    t_T = 250
    jump_length = 10
    jump_n_sample = 10

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, t_T)

    return ts


def get_schedule_jump_test(to_supplement=False):
    ts = get_schedule_jump(t_T=250, n_sample=1,
                           jump_length=10, jump_n_sample=10,
                           jump2_length=1, jump2_n_sample=1,
                           jump3_length=1, jump3_n_sample=1,
                           start_resampling=250)

    import matplotlib.pyplot as plt
    SMALL_SIZE = 8*3
    MEDIUM_SIZE = 10*3
    BIGGER_SIZE = 12*3

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.plot(ts)

    fig = plt.gcf()
    fig.set_size_inches(20, 10)

    ax = plt.gca()
    ax.set_xlabel('Number of Transitions')
    ax.set_ylabel('Diffusion time $t$')

    fig.tight_layout()

    if to_supplement:
        out_path = "/cluster/home/alugmayr/gdiff/paper/supplement/figures/jump_sched.pdf"
        plt.savefig(out_path)

    out_path = "./schedule.png"
    plt.savefig(out_path)
    print(out_path)


def main():
    get_schedule_jump_test()


if __name__ == "__main__":
    main()
