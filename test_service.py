import service
import cv2

def test_transform():
    image = cv2.imread('aviao_test.jpeg')
    image = service.transform(image)
    assert image.shape == (1, 3, 600, 600)

def test_untransform_and_draw_boxes():
    image = cv2.imread('aviao_test.jpeg')
    image = service.transform(image)
    boxes = ((0, 0, 100, 100),)
    labels = (1,)
    scores = (0.9,)
    image = service.untransform_and_draw_boxes(image[0], boxes, labels, scores)
    assert image.shape == (600, 600, 3)