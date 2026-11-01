
from pathlib import Path

import cv2
import numpy as np
from types import SimpleNamespace
import re
from ultralytics import YOLO


# ============================================================
# กำหนดตำแหน่งไฟล์โมเดล
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_CHAR_MODEL = BASE_DIR / "best.pt"
DEFAULT_PLATE_MODEL = BASE_DIR / "license_plate.pt"
MAPPING_PATH = BASE_DIR / "char_mapping.txt"


# ============================================================
# โหลดไฟล์แปลงรหัสตัวอักษร
# ============================================================

def load_char_mapping(path=MAPPING_PATH):
    mapping = {}
    path = Path(path)

    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            code, value = line.split("=", 1)
            mapping[code.strip()] = value.strip()

    return mapping


# ============================================================
# โหลดโมเดลทั้งสองตัว
# ============================================================

def load_model(
    model_path=DEFAULT_CHAR_MODEL,
    plate_model_path=DEFAULT_PLATE_MODEL
):
    char_path = Path(model_path)
    plate_path = Path(plate_model_path)

    if not char_path.is_absolute():
        char_path = (BASE_DIR / char_path).resolve()

    if not plate_path.is_absolute():
        plate_path = (BASE_DIR / plate_path).resolve()

    if not char_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดลตัวอักษร: {char_path}"
        )

    if not plate_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดลป้ายทะเบียน: {plate_path}"
        )

    return {
        "char_model": YOLO(str(char_path)),
        "plate_model": YOLO(str(plate_path)),
        "char_mapping": load_char_mapping(),
    }


# ============================================================
# อ่านชื่อคลาส
# ============================================================

def _class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))

    if isinstance(names, (list, tuple)):
        if 0 <= int(class_id) < len(names):
            return str(names[int(class_id)])

    return str(class_id)


# ============================================================
# แปลงชื่อจังหวัด
# ============================================================

def _province_name(label):
    label = str(label).strip()

    if label == "กรุงเทพมหานคร":
        return label

    if label.startswith("จังหวัด"):
        return label[len("จังหวัด"):].strip()

    return None


# ============================================================
# จัดเรียงตัวอักษรตามตำแหน่ง
# ============================================================


def _filter_overlapping_boxes(boxes, iou_threshold=0.5):
    """
    กรองกรอบตรวจจับที่ซ้อนทับกัน โดยใช้ IoU
    เก็บกรอบที่มี Confidence สูงกว่าไว้
    """

    if not boxes:
        return []

    # เรียงตาม Confidence จากมากไปน้อย
    boxes = sorted(
        boxes,
        key=lambda item: item["confidence"],
        reverse=True
    )

    filtered = []

    for box in boxes:
        overlap = False

        for existing in filtered:

            # คำนวณพื้นที่ซ้อนทับ
            x1 = max(box["x1"], existing["x1"])
            y1 = max(box["y1"], existing["y1"])
            x2 = min(box["x2"], existing["x2"])
            y2 = min(box["y2"], existing["y2"])

            intersection = (
                max(0, x2 - x1) *
                max(0, y2 - y1)
            )

            area_box = (
                (box["x2"] - box["x1"]) *
                (box["y2"] - box["y1"])
            )

            area_existing = (
                (existing["x2"] - existing["x1"]) *
                (existing["y2"] - existing["y1"])
            )

            union = area_box + area_existing - intersection

            iou = intersection / union if union > 0 else 0

            # Suppress nested duplicates only when their centers nearly agree.
            min_area = min(area_box, area_existing)
            containment = intersection / min_area if min_area > 0 else 0.0
            min_width = min(box["x2"] - box["x1"], existing["x2"] - existing["x1"])
            min_height = min(box["y2"] - box["y1"], existing["y2"] - existing["y1"])
            center_dx = abs((box["x1"] + box["x2"] - existing["x1"] - existing["x2"]) / 2)
            center_dy = abs((box["y1"] + box["y2"] - existing["y1"] - existing["y2"]) / 2)
            nested_duplicate = (
                containment >= 0.85
                and center_dx <= min_width * 0.35
                and center_dy <= min_height * 0.35
            )
            if iou > iou_threshold or nested_duplicate:
                overlap = True
                break

        if not overlap:
            filtered.append(box)

    return filtered


def _group_rows(characters):
    
    if not characters:
        return []

    rows = []

    for item in sorted(
        characters,
        key=lambda c: (c["y_center"], c["x_center"])
    ):
        item_height = max(1.0, item["y2"] - item["y1"])
        matched_row = None

        for row in rows:
            row_y = sum(
                c["y_center"] for c in row
            ) / len(row)

            row_height = sum(
                max(1.0, c["y2"] - c["y1"])
                for c in row
            ) / len(row)

            tolerance = max(
                5.0,
                min(item_height, row_height) * 0.65
            )

            if abs(item["y_center"] - row_y) <= tolerance:
                matched_row = row
                break

        if matched_row is None:
            rows.append([item])
        else:
            matched_row.append(item)

    rows.sort(
        key=lambda row: sum(
            c["y_center"] for c in row
        ) / len(row)
    )

    for row in rows:
        row.sort(key=lambda c: c["x_center"])

    return rows


# ============================================================
# ตรวจจับป้ายทะเบียนก่อน แล้วจึงอ่านตัวอักษร
# คืนค่า 4 ค่าเพื่อรองรับ app.py เดิม
# ============================================================

def _detect_plate_text_once(
    model,
    frame,
    conf_threshold=0.10,
    imgsz=1024,
    plate_padding_x=0.35,
    plate_padding_y=0.75,
    province_threshold=0.50,
    _plate_results=None
):
    annotated = frame.copy()

    if not isinstance(model, dict):
        raise ValueError("model ต้องมาจาก load_model()")

    if "char_model" not in model or "plate_model" not in model:
        raise ValueError("ไม่พบโมเดลตัวอักษรหรือโมเดลป้ายทะเบียน")

    char_model = model["char_model"]
    plate_model = model["plate_model"]
    mapping = model.get("char_mapping", {})

    # --------------------------------------------------------
    # ขั้นที่ 1: ตรวจจับกรอบป้ายทะเบียนจากภาพเต็ม
    # --------------------------------------------------------

    plate_results = _plate_results
    if plate_results is None:
        plate_results = plate_model.predict(
            frame,
            conf=0.10,
            imgsz=1280,
            max_det=20,
            verbose=False
        )

    plate_boxes = []

    for result in plate_results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = [
                int(v) for v in box.xyxy[0].tolist()
            ]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            plate_boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
            })

    # หากไม่พบป้าย ให้คืนค่าเดิมในรูปแบบที่ app.py คาดหวัง
    if not plate_boxes:
        print("ไม่พบกรอบป้ายทะเบียน")
        print("จำนวนกรอบที่ตรวจพบ: 0")
        print("จำนวนกรอบหลังกรอง: 0")
        print("ข้อความที่ตรวจพบ: ไม่พบ")
        print("จังหวัดที่ตรวจพบ: ไม่พบ")
        print("Confidence เฉลี่ย: 0.0")

        return annotated, "", 0.0, []

    # เลือกกรอบป้ายที่มี Confidence สูงสุด
    best_plate = max(
        plate_boxes,
        key=lambda item: item["confidence"]
    )

    # Expand partial plate detections; clip the crop to the source image.
    plate_width = best_plate["x2"] - best_plate["x1"]
    plate_height = best_plate["y2"] - best_plate["y1"]
    pad_x = int(round(plate_width * max(0.0, plate_padding_x)))
    pad_y = int(round(plate_height * max(0.0, plate_padding_y)))
    px1 = max(0, best_plate["x1"] - pad_x)
    py1 = max(0, best_plate["y1"] - pad_y)
    px2 = min(frame.shape[1], best_plate["x2"] + pad_x)
    py2 = min(frame.shape[0], best_plate["y2"] + pad_y)

    # วาดกรอบป้ายทะเบียน
    cv2.rectangle(
        annotated,
        (px1, py1),
        (px2, py2),
        (255, 0, 0),
        2
    )

    # ตัดภาพเฉพาะบริเวณป้ายทะเบียน
    plate_crop = frame[py1:py2, px1:px2]

    if plate_crop.size == 0:
        return annotated, "", 0.0, []

    # ขยายภาพป้ายเพื่อช่วยให้โมเดลอ่านตัวอักษร
    plate_crop = cv2.resize(
        plate_crop,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # ขั้นที่ 2: อ่านตัวอักษรจากภาพป้ายที่ตัดแล้ว
    # --------------------------------------------------------

    char_results = char_model.predict(
        plate_crop,
        conf=conf_threshold,
        imgsz=imgsz,
        max_det=100,
        verbose=False
    )

    all_candidates = []
    province_candidates = []

    crop_height, crop_width = plate_crop.shape[:2]

    for result in char_results:
        if result.boxes is None:
            continue

        print("จำนวนกรอบที่ตรวจพบ:", len(result.boxes))

        for box in result.boxes:
            x1, y1, x2, y2 = [
                float(v) for v in box.xyxy[0].tolist()
            ]

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            raw_label = _class_name(
                result.names,
                class_id
            )

            label = mapping.get(raw_label, raw_label)
            province = _province_name(label)
            if province is None:
                # Accept single Thai consonants and Arabic digits only.
                # Ignore unknown class codes and weak background detections.
                is_character = len(label) == 1 and (
                    "0" <= label <= "9" or "ก" <= label <= "ฮ"
                )
                if not is_character or confidence < max(conf_threshold, 0.25):
                    continue

            # แปลงพิกัดจากภาพป้ายกลับเป็นภาพต้นฉบับ
            original_x1 = px1 + x1 / 2
            original_y1 = py1 + y1 / 2
            original_x2 = px1 + x2 / 2
            original_y2 = py1 + y2 / 2

            item = {
                "label": label,
                "raw_label": raw_label,
                "confidence": confidence,
                "x_center": (original_x1 + original_x2) / 2,
                "y_center": (original_y1 + original_y2) / 2,
                "x1": original_x1,
                "y1": original_y1,
                "x2": original_x2,
                "y2": original_y2,
            }

            all_candidates.append(item)

            province = _province_name(label)

            if province is not None:
                province_candidates.append({
                    "label": label,
                    "province": province,
                    "province_confirmed": confidence >= province_threshold,
                    "raw_label": raw_label,
                    "confidence": confidence,
                    "x_center": (original_x1 + original_x2) / 2,
                    "y_center": (original_y1 + original_y2) / 2,
                    "x1": original_x1,
                    "y1": original_y1,
                    "x2": original_x2,
                    "y2": original_y2,
                })

    # --------------------------------------------------------
    # ขั้นที่ 3: กรองกรอบซ้ำ แล้วรวมข้อความตามตำแหน่ง
    # --------------------------------------------------------

    plate_chars = [
        item for item in all_candidates
        if _province_name(item["label"]) is None
    ]

    plate_chars = _filter_overlapping_boxes(plate_chars)

    # กรองกรอบจังหวัดที่ซ้อนกันด้วย โดยไม่ปะปนกับตัวอักษรบนป้าย
    province_candidates = _filter_overlapping_boxes(province_candidates)

    # ใช้เฉพาะกรอบที่ผ่านการกรองในการวาดและคำนวณผล
    # Return just the strongest accepted province to the app.
    province_candidates = sorted(
        province_candidates, key=lambda item: item["confidence"], reverse=True
    )[:1]
    all_candidates = plate_chars + province_candidates

    rows = _group_rows(plate_chars)

    plate_text = "".join(
        "".join(str(item["label"]) for item in row)
        for row in rows
    )

    # --------------------------------------------------------
    # ขั้นที่ 4: เลือกจังหวัดจากกรอบที่ตรวจพบ
    # --------------------------------------------------------

    best_province = None

    if province_candidates:
        best_province = max(
            province_candidates,
            key=lambda item: item["confidence"]
        )

    # --------------------------------------------------------
    # วาดกรอบตัวอักษรบนภาพต้นฉบับ
    # --------------------------------------------------------

    for item in all_candidates:
        x1 = int(max(0, item["x1"]))
        y1 = int(max(0, item["y1"]))
        x2 = int(min(frame.shape[1] - 1, item["x2"]))
        y2 = int(min(frame.shape[0] - 1, item["y2"]))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f'{item["label"]} {item["confidence"]:.2f}',
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

    if best_province:
        province_text = best_province["province"]
        province_code = best_province["raw_label"]

        cv2.putText(
            annotated,
            f'{province_code} {best_province["confidence"]:.3f}' + (
                " (uncertain)" if not best_province["province_confirmed"] else ""
            ),
            (px1, max(20, py1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2
        )

        print(
            "จังหวัดที่เลือก:",
            province_code,
            "(", province_text, ")",
            "Confidence:",
            round(best_province["confidence"], 3)
        )
    else:
        print("จังหวัดที่ตรวจพบ: ไม่พบ")

    confidences = [
        item["confidence"] for item in all_candidates
    ]

    mean_conf = (
        sum(confidences) / len(confidences)
        if confidences else 0.0
    )

    print("จำนวนกรอบหลังกรอง:", len(all_candidates))
    print("ข้อความที่ตรวจพบ:", plate_text or "ไม่พบ")
    print("Confidence เฉลี่ย:", round(mean_conf, 3))

    return annotated, plate_text, mean_conf, all_candidates


def _reading_score(reading):
    """Heuristic crop ranking, not a probability of a correct plate."""
    _, text, _, candidates = reading
    chars = [c for c in candidates if _province_name(c["label"]) is None]
    if not chars:
        return -100.0
    probabilities = [float(c["confidence"]) for c in chars]
    digits = sum("0" <= c["label"] <= "9" for c in chars)
    letters = len(chars) - digits
    # Reward additional well-supported characters, penalize weak/noisy extras.
    score = sum(p - 0.45 for p in probabilities)
    score += min(probabilities) * 0.35
    valid_layout = re.fullmatch(r"[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}", text) is not None
    if valid_layout:
        score += 0.75
    if len(chars) > 9:
        score -= (len(chars) - 9) * 0.6
    if letters > 3:
        score -= (letters - 3) * 0.5
    if digits > 4 and not valid_layout:
        score -= (digits - 4) * 0.5
    return score


def detect_plate_text(
    model,
    frame,
    conf_threshold=0.10,
    imgsz=1024,
    plate_padding_x=None,
    plate_padding_y=None,
    province_threshold=0.50,
    adaptive_crop=True,
    direct_plate=False,
):
    """Read several crop sizes; preserve the app's four-value return contract.

    Pass adaptive_crop=False for a single, faster crop in video mode.
    Low-score provinces are returned with province_confirmed=False.
    """
    if not isinstance(model, dict) or not {"char_model", "plate_model"} <= model.keys():
        raise ValueError("model must come from load_model()")
    if frame is None or frame.size == 0:
        raise ValueError("Empty input image")

    # Detect the plate once; reuse the results for each character-reading pass.
    if direct_plate:
        # Explicit close-up mode: use the entire uploaded image as the plate.
        height, width = frame.shape[:2]
        full_box = SimpleNamespace(
            conf=np.array([1.0], dtype=np.float32),
            xyxy=np.array([[0, 0, width, height]], dtype=np.float32),
        )
        plate_results = [SimpleNamespace(boxes=[full_box])]
    else:
        plate_results = model["plate_model"].predict(
            frame, conf=0.10, imgsz=1280, max_det=20, verbose=False
        )
    if not any(r.boxes is not None and len(r.boxes) for r in plate_results):
        return frame.copy(), "", 0.0, []

    if direct_plate:
        crop_sizes = [(0.0, 0.0)]
    elif plate_padding_x is not None or plate_padding_y is not None:
        crop_sizes = [(
            0.10 if plate_padding_x is None else plate_padding_x,
            0.15 if plate_padding_y is None else plate_padding_y,
        )]
    elif adaptive_crop:
        crop_sizes = [(0.0, 0.0), (0.10, 0.15), (0.35, 0.75)]
    else:
        crop_sizes = [(0.10, 0.15)]

    readings = []
    for pad_x, pad_y in crop_sizes:
        reading = _detect_plate_text_once(
            model, frame,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            plate_padding_x=pad_x,
            plate_padding_y=pad_y,
            province_threshold=province_threshold,
            _plate_results=plate_results,
        )
        readings.append(reading)

    # A tie favors the smaller crop, which was evaluated first.
    best_index = max(range(len(readings)), key=lambda i: _reading_score(readings[i]))
    print("Selected crop padding:", crop_sizes[best_index])
    return readings[best_index]