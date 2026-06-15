import os
import cv2
import numpy as np
from ultralytics import YOLO, RTDETR
from ensemble_utils import get_predictions, ensemble_with_wbf, draw_boxes

# Пути к моделям
MODEL1_PATH = "C:/Users/anyak/Desktop/Study/dipl/w-sa/an/yolo26s_china_new/yolo26s_china_new/weights/best.pt"
MODEL2_PATH = "C:/Users/anyak/Desktop/Study/dipl/w-sa/an/yolo26s_china_new_noisy/yolo26s_china_new_noisy/weights/best.pt"
MODEL3_PATH = "C:/Users/anyak/Desktop/Study/dipl/w-sa/an/rtdetr_new_china_bezmos/rtdetr_new_china_bezmos/weights/best.pt"

# Имена классов
CLASS_NAMES = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40",
    4: "Repair"
}

# Глобальные переменные для моделей (загружаются один раз)
_model1 = None
_model2 = None
_model3 = None


def load_models():
    """Загружает все модели (вызывается один раз при старте)."""
    global _model1, _model2, _model3
    print("Загрузка моделей...")
    _model1 = YOLO(MODEL1_PATH)
    _model2 = YOLO(MODEL2_PATH)
    _model3 = RTDETR(MODEL3_PATH)
    print("Все модели загружены!")


def process_image(image_path):
    """Обрабатывает изображение и возвращает результаты."""
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Предсказания
    preds1 = get_predictions(_model1, image)
    preds2 = get_predictions(_model2, image)
    preds3 = get_predictions(_model3, image)
    
    # Ансамбль с РАВНЫМИ весами и разрешением конфликтов классов
    predictions_list = [preds1, preds2, preds3]
    weights = [1.0, 1.0, 1.0]  # равные веса
    final_boxes, final_scores, final_labels = ensemble_with_wbf(
        predictions_list, (h, w), 
        iou_thr=0.5, 
        skip_box_thr=0.01, 
        weights=weights,
        resolve_conflicts=True,      # ВКЛЮЧАЕМ разрешение конфликтов
        conflict_iou_thr=0.5         # Порог IoU для конфликта
    )
    
    # Формируем список ВСЕХ боксов (для клиентской отрисовки)
    all_boxes_for_ensemble = []
    for i, (box, score, label) in enumerate(zip(final_boxes, final_scores, final_labels)):
        class_name = CLASS_NAMES.get(label, f"Class {label}")
        all_boxes_for_ensemble.append({
            'bbox': box.tolist() if hasattr(box, 'tolist') else list(box),
            'score': float(score),
            'class': int(label),
            'class_name': class_name
        })
    
    # Отрисовка для отдельных моделей
    img_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes1 = [p['bbox'] for p in preds1]
    scores1 = [p['score'] for p in preds1]
    labels1 = [p['class'] for p in preds1]
    img_model1 = draw_boxes(image.copy(), boxes1, scores1, labels1, CLASS_NAMES)
    img_model1 = cv2.cvtColor(img_model1, cv2.COLOR_BGR2RGB)
    
    boxes2 = [p['bbox'] for p in preds2]
    scores2 = [p['score'] for p in preds2]
    labels2 = [p['class'] for p in preds2]
    img_model2 = draw_boxes(image.copy(), boxes2, scores2, labels2, CLASS_NAMES)
    img_model2 = cv2.cvtColor(img_model2, cv2.COLOR_BGR2RGB)
    
    boxes3 = [p['bbox'] for p in preds3]
    scores3 = [p['score'] for p in preds3]
    labels3 = [p['class'] for p in preds3]
    img_model3 = draw_boxes(image.copy(), boxes3, scores3, labels3, CLASS_NAMES)
    img_model3 = cv2.cvtColor(img_model3, cv2.COLOR_BGR2RGB)
    
    # Для ансамбля отправляем чистое изображение (без рамок)
    img_ensemble_clean = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    # Конвертация в base64
    import base64
    from io import BytesIO
    from PIL import Image
    
    def img_to_base64(img_array):
        img = Image.fromarray(img_array)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Формируем результат
    detected_objects = []
    for i, box_info in enumerate(all_boxes_for_ensemble):
        detected_objects.append({
            'id': i + 1,
            'class': box_info['class_name'],
            'class_id': box_info['class'],
            'confidence': box_info['score'],
            'bbox': box_info['bbox']
        })
    
    # Сортируем по уверенности
    detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'original': img_to_base64(img_orig),
        'model1': img_to_base64(img_model1),
        'model2': img_to_base64(img_model2),
        'model3': img_to_base64(img_model3),
        'ensemble_clean': img_to_base64(img_ensemble_clean),
        'ensemble_boxes': all_boxes_for_ensemble,
        'detected_objects': detected_objects,
        'image_shape': [h, w],
        'stats': {
            'model1_count': len(preds1),
            'model2_count': len(preds2),
            'model3_count': len(preds3),
            'ensemble_count': len(final_boxes)
        }
    }