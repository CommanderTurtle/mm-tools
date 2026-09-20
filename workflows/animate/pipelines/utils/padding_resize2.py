import cv2
import numpy as np

def padding_resize2(img_ori, height=512, width=512, padding_color=(0, 0, 0), bbox=None, kps=None, ret_type="list"):
    """kps will be changed inplace

    """
    if bbox is not None:
        bbox = np.array(bbox)

    ori_height, ori_width, _ = img_ori.shape

    img_pad = np.zeros((height, width, 3)) + np.array(padding_color).reshape([1, 1, 3])

    if (ori_height / ori_width) > (height / width):
        new_width = int(height / ori_height * ori_width)
        img = cv2.resize(img_ori, (new_width, height), interpolation=cv2.INTER_LINEAR)
        padding = int((width - new_width) / 2)
        padding_width = padding
        padding_height = 0
        scale = height / ori_height 
        img_pad[:, padding : padding + new_width, :] = img
        if bbox is not None:
            bbox[0] = bbox[0] * new_width / ori_width + padding
            bbox[2] = bbox[2] * new_width / ori_width + padding
            bbox[1] = bbox[1] * height / ori_height
            bbox[3] = bbox[3] * height / ori_height
        if kps is not None:
            kps[:, 0] = kps[:, 0] * scale + padding
            kps[:, 1] = kps[:, 1] * scale
    else:
        new_height = int(width / ori_width * ori_height)
        img = cv2.resize(img_ori, (width, new_height), interpolation=cv2.INTER_LINEAR)
        padding = int((height - new_height) / 2)
        padding_width = 0
        padding_height = padding 
        scale = width / ori_width
        img_pad[padding : padding + new_height, :, :] = img
        if bbox is not None:
            bbox[1] = bbox[1] * new_height / ori_height + padding
            bbox[3] = bbox[3] * new_height / ori_height + padding
            bbox[0] = bbox[0] * width / ori_width
            bbox[2] = bbox[2] * width / ori_width

        if kps is not None:
            kps[:, 1] = kps[:, 1] * scale + padding
            kps[:, 0] = kps[:, 0] * scale

    img_pad = np.uint8(img_pad)


    if ret_type == "list":
        if bbox is None:
            return img_pad
        else:
            return img_pad, bbox
    elif ret_type == "dict":
        output_dict = {}
        output_dict["img"] = img_pad
        if not bbox is None:
            output_dict["bbox"] = bbox
        output_dict["padding_width"] = padding_width 
        output_dict["padding_height"] = padding_height 
        output_dict["scale"] = scale 
        return output_dict