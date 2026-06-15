import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

def calculate_iou(box1, box2):
    """Вычисляет IoU между двумя рамками в формате xyxy."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def resolve_class_conflicts(boxes, scores, labels, iou_threshold=0.5):
    """
    Разрешает конфликты, когда рамки разных классов сильно перекрываются.
    Оставляет только рамку с наибольшей уверенностью.
    
    Args:
        boxes: массив рамок [N, 4] в формате xyxy
        scores: массив уверенностей [N]
        labels: массив классов [N]
        iou_threshold: порог IoU для определения конфликта
    
    Returns:
        filtered_boxes, filtered_scores, filtered_labels
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    # Сортируем по уверенности (от большей к меньшей)
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    labels = labels[indices]
    
    keep = []
    
    for i in range(len(boxes)):
        keep_current = True
        
        for j in keep:
            # Вычисляем IoU между текущей рамкой и уже сохранённой
            iou = calculate_iou(boxes[i], boxes[j])
            
            # Если перекрытие выше порога И классы разные
            if iou >= iou_threshold and labels[i] != labels[j]:
                # Текущая рамка имеет меньшую уверенность (т.к. мы отсортировали)
                # Поэтому пропускаем её
                keep_current = False
                break
        
        if keep_current:
            keep.append(i)
    
    return boxes[keep], scores[keep], labels[keep]


def get_predictions(model, image, conf_threshold=0.25):
    """Получает предсказания от модели."""
    results = model(image, conf=conf_threshold)
    predictions = []
    
    h, w = image.shape[:2]
    
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls in zip(boxes, scores, classes):
                # ПРИНУДИТЕЛЬНО ограничиваем координаты границами изображения
                x1, y1, x2, y2 = box
                
                # Обрезаем координаты по границам изображения
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # Проверяем, что рамка имеет положительные размеры
                if x2 <= x1:
                    x2 = min(w - 1, x1 + 1)
                if y2 <= y1:
                    y2 = min(h - 1, y1 + 1)
                
                predictions.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'class': int(cls)
                })
    return predictions


def ensemble_with_wbf(predictions_list, img_shape, iou_thr=0.5, skip_box_thr=0.01, weights=None, 
                      resolve_conflicts=True, conflict_iou_thr=0.5):
    """Объединяет предсказания с помощью Weighted Boxes Fusion."""
    h, w = img_shape
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for model_predictions in predictions_list:
        boxes_norm = []
        scores = []
        labels = []
        
        for pred in model_predictions:
            x1, y1, x2, y2 = pred['bbox']
            
            # Нормализация с ПРИНУДИТЕЛЬНЫМ ограничением [0, 1]
            x1_norm = np.clip(x1 / w, 0.0, 1.0)
            y1_norm = np.clip(y1 / h, 0.0, 1.0)
            x2_norm = np.clip(x2 / w, 0.0, 1.0)
            y2_norm = np.clip(y2 / h, 0.0, 1.0)
            
            # Дополнительная проверка: ширина и высота должны быть положительными
            if x2_norm <= x1_norm:
                x2_norm = min(1.0, x1_norm + 0.001)
            if y2_norm <= y1_norm:
                y2_norm = min(1.0, y1_norm + 0.001)
            
            boxes_norm.append([x1_norm, y1_norm, x2_norm, y2_norm])
            scores.append(pred['score'])
            labels.append(pred['class'])
        
        all_boxes.append(boxes_norm)
        all_scores.append(scores)
        all_labels.append(labels)
    
    if weights is None:
        weights = [1.0] * len(predictions_list)
    
    # Вызов WBF
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        conf_type='max'
    )
    
    if len(fused_boxes) > 0:
        # Обратное преобразование с ограничением
        fused_boxes = np.array(fused_boxes) * np.array([w, h, w, h])
        fused_boxes = np.clip(fused_boxes, 0, [w-1, h-1, w-1, h-1])
        fused_boxes = fused_boxes.astype(np.int32)
        
        # Разрешаем конфликты классов
        if resolve_conflicts and len(fused_boxes) > 1:
            fused_boxes, fused_scores, fused_labels = resolve_class_conflicts(
                fused_boxes, fused_scores, fused_labels, 
                iou_threshold=conflict_iou_thr
            )
    
    return fused_boxes, np.array(fused_scores), np.array(fused_labels).astype(int)


def draw_boxes(image, boxes, scores, labels, class_names=None):
    """Рисует рамки на изображении и возвращает массив BGR."""
    img_copy = image.copy()
    
    # ЦВЕТА В BGR ДЛЯ OPENCV (порядок: B, G, R)
    colors = [
        (0, 255, 0),     # 0: D00 - зеленый
        (0, 0, 255),     # 1: D10 - красный
        (255, 0, 0),     # 2: D20 - синий
        (0, 255, 255),   # 3: D40 - желтый
        (255, 0, 255)    # 4: Repair - пурпурный
    ]
    
    if boxes is None or len(boxes) == 0:
        return img_copy
    
    for i, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            h, w = img_copy.shape[:2]
            
            # Еще раз обрезаем по границам (страховка)
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            label = int(labels[i]) if i < len(labels) else 0
            score = float(scores[i]) if i < len(scores) else 0.0
            
            color = colors[label % len(colors)]
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            if class_names:
                text = f"{class_names.get(label, label)}: {score:.2f}"
            else:
                text = f"Class {label}: {score:.2f}"
            
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Позиционирование текста (если рамка у верхнего края)
            text_y = y1 - 5
            if text_y - th < 0:
                text_y = y2 + th + 5
                cv2.rectangle(img_copy, (x1, y2), (x1 + tw, y2 + th + 5), color, -1)
                cv2.putText(img_copy, text, (x1, y2 + th), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            else:
                cv2.rectangle(img_copy, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(img_copy, text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        except Exception:
            continue
    
    return img_copy