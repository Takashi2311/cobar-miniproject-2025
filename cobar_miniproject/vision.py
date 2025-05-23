import cv2
from flygym import Fly
import numpy as np


def get_fly_vision(fly: Fly):
    assert (
        fly._curr_visual_input is not None
    ), "fly vision isn't enabled. Make sure `enable_vision` is set to True."
    return np.hstack(
        [
            fly.retina.hex_pxls_to_human_readable(
                fly._curr_visual_input[eye], True
            ).max(axis=2)[::2, ::2]
            for eye in range(2)
        ]
    )


def get_fly_vision_raw(fly: Fly):
    assert (
        fly._curr_raw_visual_input is not None
    ), "fly vision isn't enabled. Make sure `render_raw_vision` is set to True."

    return np.uint8(np.hstack(tuple(fly._curr_raw_visual_input)))


def render_image_with_vision(
    image: np.ndarray,
    vision: np.ndarray,
    odor_intensity: np.ndarray,
):
    if vision.ndim < 3:
        vision = vision[..., np.newaxis]
    if vision.shape[2] == 1:
        vision = vision.repeat(3, axis=2)

    if image.shape[1] > vision.shape[1]:
        vision = np.pad(
            vision,
            ((0,), ((image.shape[1] - vision.shape[1]) // 2,), (0,)),
            constant_values=255,
        )
    elif vision.shape[1] > image.shape[1]:
        image = np.pad(
            image,
            ((0,), ((vision.shape[1] - image.shape[1]) // 2,), (0,)),
            constant_values=255,
        )
    
    im_olf = np.full((16, image.shape[1], 3), 255, dtype=np.uint8)
    x0 = image.shape[1] // 2
    odor_intensity = odor_intensity[0] # only display the 1st odor dimension
    # average over the two types of sensors (antennae and maxillary palps)
    # and calculate the right minus left difference
    diff = odor_intensity.reshape((2, 2)).mean(0) @ (-1, 1)
    log = np.log10(np.abs(diff))
    vmin = -7
    vmax = -1
    norm = np.clip((log - vmin) / (vmax - vmin), 0, 1)
    dx = np.round(norm * x0 * np.sign(diff)).astype(int)
    cv2.rectangle(
        im_olf,
        (x0, 0),
        (x0 + dx, im_olf.shape[0]),
        (255, 127, 14),
        -1,
        cv2.LINE_AA,
    ) 
    cv2.line(
        im_olf,
        (x0, 0),
        (x0, im_olf.shape[0]),
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((vision, im_olf, image))
