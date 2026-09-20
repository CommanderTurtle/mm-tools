from copy import deepcopy


def get_padding_len(input_len, m, n, refert_num=1):

    remaining = (input_len - refert_num) % (m - refert_num)
    padding_needed = 0

    if remaining < 28:
        padding_needed = 28-remaining
    else:
        padding_needed = 4-remaining%4

    # if remaining > 0:
    #     # 计算需要填充到的目标长度L
    #     # 满足 L = k*n + 1 且 L >= remaining 的最小L
    #     # 计算最小k使得 k*n + 1 >= remaining
    #     k = max(2, (remaining - 1 + n - 1) // n)
    #     target_length = k * n + 1

    #     if target_length < 29:
    #         target_length = 29

    #     padding_needed = target_length - remaining - refert_num

    output_len = input_len + padding_needed
    return output_len


def get_padding_len_old(input_len, m, refert_num=1):

    remaining = (input_len - refert_num) % (m - refert_num)
    padding_needed = 0

    if remaining > 0:
        padding_needed = m - remaining - refert_num

    output_len = input_len + padding_needed
    return output_len


def zigzag_padding(array, target_len):
    idx = 0
    flip = False
    target_array = []
    while len(target_array) < target_len:
        target_array.append(deepcopy(array[idx]))
        if flip:
            idx -= 1
        else:
            idx += 1
        if idx == 0 or idx == len(array) - 1:
            flip = not flip
    return target_array[:target_len]


def get_valid_len(real_len, clip_len=81, overlap=1):
    real_clip_len = clip_len - overlap
    last_clip_num = (real_len - overlap) % real_clip_len
    if last_clip_num == 0:
        extra = 0
    else:
        extra = real_clip_len - last_clip_num
    target_len = real_len + extra
    return target_len


def blend_layer_with_mask(foreground, background, mask=None, border=15):
    """_summary_

    Args:
        source: 原图
        target: 目标图
        mask: 0/255 mask, [H, W, 3]
    """
    if mask is None:
        mask = np.ones_like(foreground, dtype=np.uint8) * 255

    height, width, _ = mask.shape
    border = min(border, height // 10, width // 10)

    # 防止四周全部是白的导致erode失效
    mask[:, :1] = 0
    mask[:1, :] = 0
    mask[-1:, :] = 0
    mask[:, -1:] = 0
    mask = cv2.erode(mask, np.ones([border, border]))
    mask = cv2.blur(mask, (border, border))
    mask = mask / 255
    # cv2.imwrite("mask.png", mask)
    target = (foreground * mask + background * (1 - mask)).astype(np.uint8)

    return target
