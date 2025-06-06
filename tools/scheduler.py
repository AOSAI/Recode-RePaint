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

# 1.2.2 新的完整性检查函数
def new_check_times(times, t_0, t_T, jump_length):
    assert times[-1] == -1, times[-1]  # 检查结尾是否为 -1

    for i in range(1, len(times)):
        delta = times[i] - times[i - 1]
        assert delta == -1 or delta == jump_length - 1, \
            f"Invalid jump step at index {i}: {times[i-1]} → {times[i]})"

    for t in times:
        assert t_0 <= t <= t_T, f"Timestep {t} out of bounds [{t_0}, {t_T}]"

# 1.2.1 回跳的条件判断的封装函数
def judge_bool(t, jump_list, jump_length, start_resampling=10000):
    if t not in jump_list:
        return False
    if t > start_resampling - jump_length:
        return False
    if jump_list[t] < 1:
        return False
    return True

# 1.1.1 构造跳跃字典
def build_jumps(start=250, end=0, length=10, count=10):
    """
    :range(0, 240, 10) -> [0, 10, 20, 30 ...]
    :dict {0: 9, 10: 9, 20: 9, ..., 230: 9}
    """
    return {j: count - 1 for j in range(start, end, -length)}


# ------------ 2. RePaint+ 的 jump back 跳跃时间表 ------------
def get_schedule_jump2(
    t_T, jump_length, jump_sapcing, jump_n_sample, add_noise_once,
    jump_start_pct=0.2, jump_stop_pct=0.8, 
):
    # 2.1 获取 jump back 的时间步范围
    start_jump = t_T - int(t_T * jump_start_pct)
    stop_jump = t_T - int(t_T * jump_stop_pct)
    
    # 2.2 结构为 3 个数组；计算重采样的节点和次数 
    ts_before_start = list(range(t_T - 1, start_jump - 1, -1))
    ts_after_stop = list(range(stop_jump - 1, -2, -1))
    jumps  = build_jumps(start_jump, stop_jump, jump_sapcing, jump_n_sample)

    ts = []
    t = start_jump
    while t > stop_jump:
        if judge_bool(t, jumps, jump_length):
            jumps[t] -= 1
            if not add_noise_once:
                for i in range(1, jump_length):
                    t = t + 1
                    ts.append(t)
            else:
                t = t + jump_length - 1
                ts.append(t)
        t = t - 1
        ts.append(t)

    ts = ts_before_start + ts + ts_after_stop
    if not add_noise_once:
        _check_times(ts, -1, t_T)  
    else: 
        new_check_times(ts, -1, t_T, jump_length)
    return ts


# ------------ 1. RePaint 的核心，jump back 跳跃时间表（已重构） ------------
def get_schedule_jump(
    t_T, n_sample, jump_length, jump_n_sample, add_noise_once, jump2_length=1, 
    jump2_n_sample=1, jump3_length=1, jump3_n_sample=1, start_resampling=100000
):
    # 1.1 从 t_T 到 0，每隔 jump_length 步设置一个“回跳点”，重采样 jump_n_sample - 1 次
    jumps  = build_jumps(t_T, 0, jump_length, jump_n_sample)
    jumps2 = build_jumps(t_T, 0, jump2_length, jump2_n_sample)
    jumps3 = build_jumps(t_T, 0, jump3_length, jump3_n_sample)

    # 1.2 循环跳跃，重构时间步
    t = t_T  # 以 t_T = 250 为例子
    ts = []  # 跳跃时间表
    while t >= 1:
        t = t - 1
        ts.append(t)

        # 每个时间点重采样 n_sample 次，为 1 时等于跳过，不做这一步
        if (t + 1 < t_T - 1 and t <= start_resampling):
            for _ in range(n_sample - 1):
                ts.append(t + 1)
                ts.append(t)

        # jump3：更细粒度的回跳（扩展接口）
        if judge_bool(t, jumps3, jump3_length, start_resampling):
            jumps3[t] -= 1
            for i in range(1, jump3_length):
                t = t + 1
                ts.append(t)

        # jump2：中级回跳 + 重置 jump3（扩展接口）
        if judge_bool(t, jumps2, jump2_length, start_resampling):
            jumps2[t] -= 1
            for i in range(1, jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = build_jumps(t_T, 0, jump3_length, jump3_n_sample)

        # jump：最粗粒度回跳 + 重置 jump2 & jump3（实现）
        if judge_bool(t, jumps, jump_length, start_resampling):
            jumps[t] -= 1
            if not add_noise_once:
                for i in range(1, jump_length):
                    t = t + 1
                    ts.append(t)
            else:
                t = t + jump_length - 1
                ts.append(t)
            jumps2 = build_jumps(t_T, 0, jump2_length, jump2_n_sample)
            jumps3 = build_jumps(t_T, 0, jump3_length, jump3_n_sample)

    ts.append(-1)  # 模型会在 -1 时结束循环
    # 断言检查函数，确保所有 ts 都合法
    if not add_noise_once:
        _check_times(ts, -1, t_T)  
    else: 
        new_check_times(ts, -1, t_T, jump_length)
    return ts

if __name__ == "__main__":
    # result = get_schedule_jump(
    #     t_T=250, n_sample=1, jump_length=10, jump_n_sample=10, add_noise_once=True
    # )
    # print(result)

    result2 = get_schedule_jump2(
        t_T=250, jump_length=10, jump_sapcing=10, jump_n_sample=10, add_noise_once=False,
        jump_start_pct=0.04, jump_stop_pct=1.0
    )
    print(result2)

    