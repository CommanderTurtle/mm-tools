import cv2
import math
import numpy as np


def calculate_new_size(orig_w, orig_h, target_area, divisor=64):
    """
    计算满足条件的新图片尺寸，保持最接近原始长宽比

    参数:
        orig_w (int): 原始图片宽度
        orig_h (int): 原始图片高度
        target_area (int): 目标面积上限

    返回:
        tuple: (new_w, new_h) 新的宽度和高度
    """

    target_ratio = orig_w / orig_h

    def check_valid(w, h):
        """检查尺寸是否满足所有条件"""
        if w <= 0 or h <= 0:
            return False
        return (w * h <= target_area and  # 面积条件
                w % divisor == 0 and  # 宽度是64的倍数
                h % divisor == 0)  # 高度是64的倍数

    def get_ratio_diff(w, h):
        """计算与目标宽高比的差异"""
        return abs(w / h - target_ratio)

    def round_to_64(value, round_up=False, divisor=64):
        """将数值调整为最接近的64的倍数"""
        if round_up:
            return divisor * ((value + (divisor - 1)) // divisor)
        return divisor * (value // divisor)

    # 生成可能的尺寸组合
    possible_sizes = []

    # 计算理论最大值
    max_area_h = int(np.sqrt(target_area / target_ratio))
    max_area_w = int(max_area_h * target_ratio)

    # 确保最大边界是64的倍数
    max_h = round_to_64(max_area_h, round_up=True, divisor=divisor)
    max_w = round_to_64(max_area_w, round_up=True, divisor=divisor)

    # 遍历所有可能的高度（64的倍数）
    for h in range(divisor, max_h + divisor, divisor):
        # 根据当前高度和目标比例计算理想宽度
        ideal_w = h * target_ratio

        # 尝试向上和向下取整到64的倍数
        w_down = round_to_64(ideal_w)
        w_up = round_to_64(ideal_w, round_up=True)

        # 检查两个可能的宽度
        for w in [w_down, w_up]:
            if check_valid(w, h, divisor):
                possible_sizes.append((w, h, get_ratio_diff(w, h)))

    # 如果没有找到有效尺寸，抛出异常
    if not possible_sizes:
        raise ValueError("无法找到满足条件的尺寸")

    # 按面积（降序）和比例差异（升序）排序
    possible_sizes.sort(key=lambda x: (-x[0] * x[1], x[2]))

    # 返回面积最大且比例最接近的组合
    best_w, best_h, _ = possible_sizes[0]
    return int(best_w), int(best_h)


def resize_by_area(image, target_area, keep_aspect_ratio=True, divisor=16, padding_color=(0, 0, 0), return_padding_info=False):
    h, w = image.shape[:2]
    try:
        new_w, new_h = calculate_new_size(w, h, target_area, divisor)
    except:
        aspect_ratio = w / h

        if keep_aspect_ratio:
            # 保持宽高比，计算新尺寸
            new_h = math.sqrt(target_area / aspect_ratio)
            new_w = target_area / new_h
        else:
            # 不保持宽高比，直接取平方根（假设为正方形）
            new_w = new_h = math.sqrt(target_area)

        new_w, new_h = int((new_w // divisor) * divisor), int((new_h // divisor) * divisor)

    # 调整尺寸（使用 INTER_AREA 插值缩小，INTER_LINEAR 放大）
    interpolation = cv2.INTER_AREA if (new_w * new_h < w * h) else cv2.INTER_LINEAR

    # resized_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    resized_image = padding_resize(image, height=new_h, width=new_w, padding_color=padding_color,
                                    interpolation=interpolation, return_padding_info=return_padding_info)  
    return resized_image

    
def padding_resize(img_ori, height=512, width=512, padding_color=(0, 0, 0), interpolation=cv2.INTER_LINEAR, return_padding_info=False):
    ori_height = img_ori.shape[0]
    ori_width = img_ori.shape[1]
    channel = img_ori.shape[2]

    img_pad = np.zeros((height, width, channel))
    if channel == 1:
        img_pad[:, :, 0] = padding_color[0]
    else:
        img_pad[:, :, 0] = padding_color[0]
        img_pad[:, :, 1] = padding_color[1]
        img_pad[:, :, 2] = padding_color[2]

    if (ori_height / ori_width) > (height / width):
        new_width = int(height / ori_height * ori_width)
        img = cv2.resize(img_ori, (new_width, height), interpolation=interpolation)
        padding = int((width - new_width) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]  # 如果是灰度图，添加一个通道维度
        img_pad[:, padding: padding + new_width, :] = img
        padding_type = 'width'
        side_long = new_width
    else:
        new_height = int(width / ori_width * ori_height)
        img = cv2.resize(img_ori, (width, new_height), interpolation=interpolation)
        padding = int((height - new_height) / 2)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]  # 如果是灰度图，添加一个通道维度
        img_pad[padding: padding + new_height, :, :] = img
        padding_type = 'height'
        side_long = new_height

    img_pad = np.uint8(img_pad)
    if return_padding_info:
        padding_info = {
            'padding_type': padding_type,
            "padding": padding,
            'side_long': side_long
        }
        return img_pad, padding_info
    else:
        return img_pad


def get_frame_indices(frame_num, video_fps, clip_length, train_fps):
    """
    计算视频帧采样的帧号列表，当需要的帧数超过实际帧数时，通过正序-倒序-正序方式填充

    参数:
        frame_num (int): 视频总帧数
        video_fps (float): 视频帧率
        clip_length (int): 需要采样的帧数
        train_fps (float): 目标采样帧率

    返回:
        list: 采样帧号列表
    """
    # 正常情况：所需帧数小于等于总帧数
    start_frame = 0
    times = np.arange(0, clip_length) / train_fps
    frame_indices = start_frame + np.round(times * video_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, frame_num - 1)

    return frame_indices.tolist()

