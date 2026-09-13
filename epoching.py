# 三个不同实验的实验过程与采集后信号的切段介绍：
#三个实验都用VR设备对被试进行刺激，采集脑电信号。用于具体的实验课题分析

# 实验一：手写汉字的不同笔画脑电信号分类实验
#实验将给被试呈现不同的汉字书写过程（推荐被试跟着写），每一个笔画设计为3.5s的书写时间，
#不同的笔画/汉字将会重复播放若干遍，组间间隔若干遍让被试休息
#为了确定一组信号的起始采集时间和结束采集时间，将会由VR设备发出标记信号（幅值相比与神经信号大得多）作为标记，方便不同笔画/汉字的脑电信号切段。
#通过VR标记信号进行切分把相关神经信号分为“横”“竖”“撇”“捺”“点”“其他”六类，方便后续的脑电信号分析和分类。

# 实验二：情绪神经信号分析实验
#被试将观看四段恐怖类的VR视频主题与“深海”，“电梯”等相关
#每段视频播放和结束时刻都会由VR设备发出标记信号，方便后续的脑电信号切段。
#将四段视频的脑电信号切分为“V_49”“V_92”“V_212”“V_208”四类（命名由视频长度决定），方便后续的脑电信号分析和分类。

# 实验三：不同种类物品的视觉刺激神经信号分类实验
#被试将观看三个完整视频，每个视频由不同种类的物品片段组成，每个物品片段播放和结束时刻都会由VR设备发出标记信号，方便后续的脑电信号切段。

#三个实验不仅采集了脑电信号，还采集了心电信号

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_closing  # 新增 binary_closing 填小间隙

#读取有HL_2版本数据，和无HL_2版本数据（部分数据实验时没有采集HL_2信号）
with_hl2_path = r""   # 带 HL_2 的文件
no_hl2_path   = r""               # 无 HL_2 的文件（EEGLAB 手动保存的）

# 读取数据
raw = mne.io.read_raw_eeglab(with_hl2_path, preload=True)
data, times = raw['HL 2']
data = data.flatten()
sfreq = raw.info['sfreq']

# ==================== 智能双模式参数（改进版，防块内小间隙多标）====================
baseline = -5550e-6  # 静息基线
relative_thresh = 2200e-6  # 相对阈值（4.4 mV，抓得早又稳）

gap_fill_sec = 0.06  # 填补 <60 ms 间隙（防 50 ms 降幅导致断块）

short_block_sec = 0.6  # 400 ms 信号块长度阈值（<0.6 s 算短块）
long_block_sec = 0.8  # 1 s 信号块长度阈值（>0.8 s 算长块）

short_min_gap_sec = 0.3  # 短块之间最小间隔（允许 400 ms 间隔出现两个）
long_min_gap_sec = 2.0  # 长块之间最小间隔（保持 2 s 死区）
# ====================================================

# 相对基线偏差
deviation = np.abs(data - baseline)
above = deviation > relative_thresh

# 改进1：填补块内小间隙（<60 ms 降幅视为同一个块）
gap_fill_samples = int(gap_fill_sec * sfreq)
above = binary_closing(above, structure=np.ones(gap_fill_samples))  # 连接小间隙

# 改进2：用 label 找连续块
labeled_blocks, num_blocks = label(above)
valid_triggers = []

for block_id in range(1, num_blocks + 1):
    block_samples = np.where(labeled_blocks == block_id)[0]
    if len(block_samples) < 5:  # 太短噪声忽略
        continue

    start_idx = block_samples[0]  # 块第一个点
    block_duration_sec = (block_samples[-1] - start_idx) / sfreq

    # 短/长块选择死区
    if block_duration_sec > long_block_sec:
        min_gap_sec = long_min_gap_sec
    else:
        min_gap_sec = short_min_gap_sec

    min_gap_samples = int(min_gap_sec * sfreq)

    # 如果和上一个足够远，就记录
    if not valid_triggers or (start_idx - valid_triggers[-1] >= min_gap_samples):
        valid_triggers.append(start_idx)

valid_triggers = np.array(valid_triggers, dtype=int)
trigger_times = times[valid_triggers]

# 输出结果
print(f"智能双模式检测完成（改进版：填补50ms间隙，防首尾多标）！")
print(f"   共检测到 {len(trigger_times)} 个事件")
print(f"   长信号 (>0.8s) 用 2s 死区，短信号 (<0.6s) 用 0.3s 死区")
print("所有时间戳：", np.round(trigger_times[0:len(trigger_times)], 4))

# 可视化
plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.plot(times, data * 1e6, 'k-', lw=0.8)
plt.axhline(baseline * 1e6, color='blue', linestyle='--', label='基线')
plt.plot(trigger_times, data[valid_triggers] * 1e6, 'o', color='orange', markersize=8, label='触发点')

plt.title('改进版：块内50ms降幅视为一个信号（首尾只标一个）')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#Matlab里更新后会保存为v7.3版本，需要通过% 关键：强制用旧格式保存
# EEG = pop_saveset(EEG, 'filename', 'Acq 2025_07_30_1456_fianl.set', 'filepath', 'E:\XieBro\分任务切段data\2025_7_30多个数据\Acq 2025_07_30_1456', 'version', '7');



# #分段
# import os
# from pathlib import Path
#
# # ------------------ 加载无 HL_2 的干净 EEG 数据 ------------------
# print("\n正在加载无 HL_2 的干净 EEG 数据用于分段...")
# raw_clean = mne.io.read_raw_eeglab(no_hl2_path, preload=True)
# print(f"干净数据加载完成，采样率: {raw_clean.info['sfreq']} Hz, 通道数: {len(raw_clean.ch_names)}")
#
# # ------------------ 检查触发点数量 ------------------
# #上面代码通过可视化
#
# # ------------------ 设置输出目录与类别文件夹 ------------------
# output_root = r"E:\XieBro\分任务切段data\2025_7_31至8_6数据\Acq 2025_08_02_1707\split_segments"
# Path(output_root).mkdir(parents=True, exist_ok=True)
#
# selected_categories = ["V1","V2","V3","V4","V5","V6"]
# #手动设置输出文件夹
# # 选项1：实验一（笔画分类）
# # selected_categories = ["横", "竖", "撇", "捺", "点", "其他"]
#
# # 选项2：实验二（情绪神经信号分析实验）
# # selected_categories = ["V_49", "V_92", "V_212", "V_208"]
#
# # 选项3：实验三（不同种物品的视觉刺激神经信号分类实验）
# # selected_categories = ["V1","V2","V3","V4","V5","V6"]
#
# # 自动创建所有选择的类别文件夹
# print(f"正在创建以下类别文件夹（共 {len(selected_categories)} 个）：")
# for cat in selected_categories:
#     cat_dir = Path(output_root) / cat
#     cat_dir.mkdir(exist_ok=True)
#     print(f"   已创建/存在：{cat_dir}")
#
# print(f"\n所有输出目录准备完成！根目录：{output_root}")
#
#
# # ------------------  辅助函数：均分区间并保存小片段 ------------------
# def split_and_save(start_time, end_time, n_segments, category_sequence):
#     """
#     将 [start_time, end_time) 均分成 n_segments 份
#     category_sequence: 可以是字符串（如 "横"）或列表（如循环序列）
#     """
#     if n_segments <= 0:
#         return
#     segment_duration = (end_time - start_time) / n_segments
#     for i in range(n_segments):
#         tmin = start_time + i * segment_duration
#         tmax = start_time + (i + 1) * segment_duration
#
#         # 裁剪出一个小片段
#         epoch = raw_clean.copy().crop(tmin=tmin, tmax=tmax)
#
#         # 确定类别
#         if isinstance(category_sequence, str):
#             cat = category_sequence
#         else:
#             cat = category_sequence[i]
#
#         # 保存路径与文件名：改成 “类别名_序号.fif”
#         save_dir = os.path.join(output_root, cat)
#         existing_files = len([f for f in os.listdir(save_dir) if f.endswith('.fif')])  # 只数 .fif 文件
#         filename = f"{cat}_{existing_files:04d}.fif"  # ← 关键！类别名 + 4位序号
#         save_path = os.path.join(save_dir, filename)
#
#         epoch.save(save_path, overwrite=True)
#
#
# # ------------------ 执行你的分段规则 ------------------
# t = trigger_times  # 简写，t[0] 是第1个触发点，t[1] 是第2个...
# print("\n开始按规则分段并保存...")
#
# # #实验一：汉字
# # # 1→2 舍弃
# # # 2→3 均分25份 → "横"
# # split_and_save(t[1], t[2], 25, "横")
# #
# # # 3→4 舍弃
# # # 4→5 均分25份 → "竖"
# # split_and_save(t[3], t[4], 25, "竖")
# #
# # # 5→6 舍弃
# # # 6→7 均分25份 → "撇"
# # split_and_save(t[5], t[6], 25, "撇")
# #
# # # 7→8 舍弃
# # # 8→9 均分25份 → "捺"
# # split_and_save(t[7], t[8], 25, "捺")
# #
# # # 9→10 舍弃
# # # 10→11 均分25份 → "点"
# # split_and_save(t[9], t[10], 25, "点")
# #
# # # 11→12 舍弃
# # # 12→13 均分40份，循环：横→竖→撇→捺（重复10次）
# # cycle_1 = ["横", "竖", "撇", "捺"] * 10
# # split_and_save(t[11], t[12], 40, cycle_1)
# #
# # # 13→14 舍弃
# # # 14→15 均分90份，自定义循环顺序（重复10次）
# # cycle_2 = ["撇", "横", "竖", "撇", "点", "点", "点", "横", "竖"] * 10
# # split_and_save(t[13], t[14], 90, cycle_2)
# #
# # # 15→16 舍弃
# # # 16→17 均分10份，（字：人，儿，入，八，十）
# # cycle_3 = ["撇", "捺", "撇", "其他", "撇", "捺", "撇", "捺", "横", "竖"]
# # split_and_save(t[15], t[16], 10, cycle_3)
# #
# # # 17-18舍弃
# # #18-19 均分30份，（字：万，丈，上，下，与，么，义，之，亡，千）
# # cycle_4=["横","其他","撇","横","撇","捺","竖","横","横","横","竖","点","横","其他","横","撇","其他","点","点","撇","捺","点","其他","捺","点","横","其他","撇","横","竖"]
# # split_and_save(t[17],t[18],30,cycle_4)
# #
# # #19-20舍弃
# # #20-21均分80份，（字：不，丑，中，丰，为，云，五，井，什，仁，仆，今，介，从，公，六，分，区，升，午）
# # cycle_5=["横","撇","竖","点","其他","竖","横","横","竖","其他","横","竖","横","横","横","竖","点","撇","其他","点",
# #          "横","横","其他","点","横","竖","其他","横","横","横","撇","竖","撇","竖","横","竖","撇","竖","横","横",
# #          "撇","竖","竖","点","撇","捺","点","其他","撇","捺","撇","竖","撇","点","撇","捺","撇","捺","其他","点",
# #          "点","横","撇","点","撇","捺","其他","撇","横","撇","点","其他","撇","横","撇","竖","撇","横","横","竖"]
# # split_and_save(t[19],t[20],80,cycle_5)
# #
# # #21-22舍弃
# # #22-23均分50份，（字：且，世，业，丘，丙，主，仙，令，仪，刊）
# # cycle_6=["竖","其他","横","横","横","横","竖","竖","横","其他","竖","竖","点","撇","横","撇","竖","横","竖","横",
# #          "横","竖","其他","撇","点","点","横","横","竖","横","撇","竖","竖","其他","竖","撇","捺","点","其他","点",
# #          "撇","竖","点","撇","捺","横","横","竖","竖","其他"]
# # split_and_save(t[21],t[22],50,cycle_6)
# #
# # #23-24舍弃
# # #24-25均分成120份，（字：亥，仲，价，任，份，仿，企，伊，伍，伏，伐，众，会，伟，伤，伦，伪，全，关，兴）
# # cycle_7=["点","横","其他","撇","撇","点","撇","竖","竖","其他","横","竖","撇","竖","撇","捺","撇","竖","撇","竖",
# #          "撇","横","竖","横","撇","竖","撇","捺","其他","撇","撇","竖","点","横","其他","撇","撇","捺","竖","横",
# #          "竖","横","撇","竖","其他","横","横","撇","撇","竖","横","竖","其他","横","撇","竖","横","撇","捺","点",
# #          "撇","竖","横","其他","撇","点","撇","捺","撇","点","撇","捺","撇","捺","横","横","其他","点","撇","竖",
# #          "横","横","其他","竖","撇","竖","撇","横","其他","撇","撇","竖","撇","捺","撇","其他","撇","竖","点","撇",
# #          "其他","点","撇","捺","横","横","竖","横","点","撇","横","横","撇","捺","点","点","撇","横","撇","点"]
# # split_and_save(t[23],t[24],120,cycle_7)
# #
# # #25-26舍弃
# # #26-27均分56份，（字：亩，伸，但，位，住，佐，体，作）
# # cycle_8=["点","横","竖","其他","横","竖","横","撇","竖","竖","其他","横","横","竖","撇","竖","竖","其他","横","横",
# #          "横","撇","竖","点","横","点","撇","横","撇","竖","点","横","横","竖","横","撇","竖","横","撇","横",
# #          "竖","横","撇","竖","横","竖","撇","捺","横","撇","竖","撇","横","竖","横","横"]
# # split_and_save(t[25],t[26],56,cycle_8)
# #
# # #27-28舍弃
# # #28-29均分16份：（字：佳，侩）
# # cycle_9=["撇","竖","横","竖","横","横","竖","横","撇","竖","撇","捺","横","横","其他","点"]
# # split_and_save(t[27],t[28],16,cycle_9)
# #
# # #29-30舍弃
# # #30-31均分18份，（字：保，信）
# # cycle_10=["撇","竖","竖","其他","横","横","竖","撇","捺","撇","竖","点","横","横","横","竖","其他","横"]
# # split_and_save(t[29],t[30],18,cycle_10)
#
#
# # #实验二：观看恐怖视频
# # split_and_save(t[2], t[3], 1, "V_212")
# # split_and_save(t[5], t[6], 1, "V_49")
# # split_and_save(t[4], t[5], 1, "V_92")
# # split_and_save(t[6], t[7], 1, "V_49")
#
# #实验三：观看多种类型视频片段
# split_and_save(t[1], t[2], 1, "V1")
# split_and_save(t[4], t[5], 1, "V2")
# split_and_save(t[7], t[8], 1, "V3")
# split_and_save(t[10], t[11], 1, "V4")
# split_and_save(t[13], t[14], 1, "V5")
# split_and_save(t[16], t[17], 1, "V6")
#
#
#
#
# # #辅助函数
# # def split_and_save_with_skip(start_time, end_time, n_segments, category_sequence, skip_sec=3.0):
# #     if n_segments <= 0:
# #         return
# #
# #     total_duration = end_time - start_time
# #     segment_duration = total_duration / n_segments  # 每份总长度
# #
# #     keep_duration = segment_duration - skip_sec  # 每份保留长度（前 skip_sec 秒舍去）
# #
# #     if keep_duration <= 0:
# #         print(f"警告：每份总长度 {segment_duration:.3f}s，小于 skip_sec {skip_sec}s，跳过此区间")
# #         return
# #
# #     for i in range(n_segments):
# #         seg_start = start_time + i * segment_duration  # 本份开始时间
# #         seg_end = seg_start + segment_duration  # 本份结束时间
# #
# #         tmin = seg_start + skip_sec  # 舍去前 3s，从这里开始
# #         tmax = seg_end  # 到本份结束
# #
# #         # 裁剪
# #         epoch = raw_clean.copy().crop(tmin=tmin, tmax=tmax)
# #
# #         # 确定类别
# #         if isinstance(category_sequence, str):
# #             cat = category_sequence
# #         else:
# #             cat = category_sequence[i]
# #
# #         # 保存：类别名_序号.fif
# #         save_dir = os.path.join(output_root, cat)
# #         existing_files = len([f for f in os.listdir(save_dir) if f.endswith('.fif')])
# #         filename = f"{cat}_{existing_files:04d}.fif"
# #         save_path = os.path.join(save_dir, filename)
# #
# #         epoch.save(save_path, overwrite=True)
# #         print(f"   已保存（舍去前{skip_sec}s）: {filename} → {cat} 文件夹")
#
# # #v1的顺序
# # cycle_11=["建筑", "海岸沙滩", "水母","兔子","滑雪","树林","熊猫","摩托车","道路","面容","香蕉","钢琴","大象","舞蹈","山","船/艇","拳击","烟花","热气球","架子鼓",
# #  "吉他","马","飞机","跑步","猫","河流/瀑布","双人交谈","蛋糕","蘑菇","西瓜","汽车","狗","海龟/乌龟","花朵","饮品/酒水","披萨","鱼群/珊瑚","人群","鸟类","鲨鱼/鲸"]
# # #v2的顺序
# # cycle_12=["烟花","架子鼓","海岸沙滩","香蕉","披萨","花朵","热气球","马","滑雪","跑步","钢琴","飞机","船/艇","熊猫","山","西瓜","拳击","蘑菇","狗","舞蹈",
# #           "吉他","道路","面容","饮品/酒水","建筑","大象","鱼群/珊瑚","蛋糕","鸟类","双人交谈","海龟/乌龟","树林","汽车","兔子","人群","猫","鲨鱼/鲸","摩托车","河流/瀑布","水母",]
# # #v3的顺序
# # cycle_13=["拳击","摩托车","披萨","猫","吉他","大象","汽车","花朵","马","熊猫","人群","山","树林","舞蹈","飞机","双人交谈","香蕉","蛋糕","滑雪","西瓜",
# #           "狗","烟花","鱼群/珊瑚","面容","蘑菇","鲨鱼/鲸","饮品/酒水","船/艇","跑步","河流/瀑布","海龟/乌龟","水母","架子鼓","道路","钢琴","鸟类","热气球","海岸沙滩","建筑","兔子",]
# # split_and_save(t[5], t[6], 40, cycle_11)
#
#
# print("\n保存完成！")