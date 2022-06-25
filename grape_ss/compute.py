

def intersection_over_union(boxes_preds_a, boxes_labels_a):

    res = 0

    for boxes_preds in boxes_preds_a:

        lst = []

        for boxes_labels in boxes_labels_a:

            box1_x1 = boxes_preds[0] - boxes_preds[2] / 2
            box1_y1 = boxes_preds[1] - boxes_preds[3] / 2
            box1_x2 = boxes_preds[0] + boxes_preds[2] / 2
            box1_y2 = boxes_preds[1] + boxes_preds[3] / 2
            box2_x1 = boxes_labels[0] - boxes_labels[2] / 2
            box2_y1 = boxes_labels[1] - boxes_labels[3] / 2
            box2_x2 = boxes_labels[0] + boxes_labels[2] / 2
            box2_y2 = boxes_labels[1] + boxes_labels[3] / 2

            x1 = max(box1_x1, box2_x1)
            y1 = max(box1_y1, box2_y1)
            x2 = min(box1_x2, box2_x2)
            y2 = min(box1_y2, box2_y2)

            # 최소값을 0으로 설정
            x21 = max(x2 - x1, 0)
            y21 = max(y2 - y1, 0)
            intersection = x21 * y21

            box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
            box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

            lst.append(intersection / (box1_area + box2_area - intersection + 1e-6))

        res += max(lst)

    res /= max(len(boxes_preds_a), 1)

    return res
