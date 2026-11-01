import os
import hashlib
import io

import tempfile

import time

from pathlib import Path

from datetime import datetime



import cv2

import pandas as pd
import numpy as np

import streamlit as st

from PIL import Image, ImageOps



import database as db

from plate_detector import load_model, detect_plate_text



# ============================================================

# ตั้งค่า: แก้ MODEL_PATH ให้ตรงกับ best.pt ของโปรเจกต์เดิม

# ตัวอย่าง:

# "runs/detect/runs/baseline-3/weights/best.pt"

# "runs/detect/runs/tuned/weights/best.pt"

# หรือใส่ path เต็ม เช่น r"C:\\\Users\\\usEr\\\Downloads\\\test_ml\\\best.pt"

# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"



st.set_page_config(page_title="ระบบจัดการรถเข้าออกหมู่บ้าน", page_icon="🚗", layout="wide")

db.init_db()





@st.cache_resource

def get_yolo_model(path):

          return load_model(path)





def normalize_for_match(text):

          return db.normalize_plate(text)


def province_from_candidates(candidates):
    """อ่านชื่อจังหวัดจาก label ภาษาไทยที่ plate_detector แปลงจาก char_mapping.txt แล้ว"""
    for item in candidates or []:
        if not item.get("province_confirmed", True):
            continue
        label = str(item.get("label", "")).strip()
        if label == "กรุงเทพมหานคร":
            return label
        if label.startswith("จังหวัด"):
            return label[len("จังหวัด"):].strip()
    return None


def check_gate(plate_text, candidates=None):
    """อนุญาตเมื่อทะเบียนและจังหวัดตรงกับลูกบ้านที่ active เท่านั้น"""
    province = province_from_candidates(candidates)
    if not plate_text or not province:
        return None, False, "ปิด"
    resident = db.find_authorized_resident(plate_text, province)
    if resident:
        return resident, True, "เปิด (จำลอง)"
    return None, False, "ปิด"


def write_event_once(plate_text, confidence, source, detail, state, cooldown=30):

          """Avoid creating a log row for the same plate every frame."""

          key = f"{source}:{normalize_for_match(plate_text)}"

          now = time.time()

          last = st.session_state.get("_last_event_times", {})

          if not plate_text or now - last.get(key, 0) < cooldown:

                    return

          resident, authorized, gate_status = check_gate(plate_text)

          db.add_vehicle_log(

                    plate_text=plate_text,

                    resident=resident,

                    authorized=authorized,

                    gate_status=gate_status,

                    source=source,

                    confidence=confidence,

                    detail=detail

          )

          last[key] = now

          st.session_state["_last_event_times"] = last





def resident_frame(rows):

          columns = ["id", "owner_name", "plate_text", "province", "house_no", "phone", "note", "active", "created_at"]

          return pd.DataFrame([dict(row) for row in rows], columns=columns)





def show_dashboard():

          rows = db.list_residents(include_inactive=True)

          logs = db.list_vehicle_logs(limit=5000)

          active = sum(1 for row in rows if row["active"])

          authorized_today = 0

          denied_today = 0

          today = datetime.now().date().isoformat()

          for row in logs:

                    if str(row["event_time"]).startswith(today):

                              if row["authorized"]:

                                        authorized_today += 1

                              else:

                                        denied_today += 1



          c1, c2, c3, c4 = st.columns(4)

          c1.metric("ทะเบียนทั้งหมด", len(rows))

          c2.metric("ทะเบียนที่ใช้งาน", active)

          c3.metric("รถที่ผ่านวันนี้", authorized_today)

          c4.metric("รถที่ไม่ผ่านวันนี้", denied_today)

          st.info("สถานะประตูเป็นการจำลองบนเว็บเท่านั้น ระบบนี้ยังไม่ได้เชื่อมต่ออุปกรณ์เปิดประตูจริง")





def resident_management():
    st.subheader("เพิ่มข้อมูลลูกบ้าน / รถ")

    # อัปโหลดภาพป้ายทะเบียน
    st.markdown("#### เพิ่มข้อมูลจากภาพป้ายทะเบียน")
    image_upload = st.file_uploader(
        "เลือกรูปภาพรถหรือป้ายทะเบียน",
        type=["png", "jpg", "jpeg", "webp"],
        key="resident_plate_image",
        help="หลังอ่านภาพแล้วสามารถแก้ไขข้อมูลทุกช่องก่อนบันทึกได้",
    )

    if image_upload is not None:
        # ใช้ bytes ของรูปที่อัปโหลดโดยตรง เพื่อให้ได้ข้อมูลของรูปปัจจุบัน
        image_bytes = image_upload.getvalue()
        image_signature = hashlib.md5(image_bytes).hexdigest()

        # ถ้าเปลี่ยนรูป ให้ล้างผลอ่านจากรูปเก่าก่อน
        if st.session_state.get("resident_image_signature") != image_signature:
            st.session_state["resident_image_signature"] = image_signature
            for key in (
                "resident_ocr_text",
                "resident_ocr_conf",
                "resident_ocr_preview",
                "resident_ocr_message",
                "resident_province_suggestion",
            ):
                st.session_state.pop(key, None)

            # ล้างเฉพาะทะเบียนและจังหวัดที่ได้จาก OCR ก่อนหน้า
            st.session_state["new_plate"] = ""
            st.session_state["new_province"] = "ไม่ระบุ"

        st.image(image_bytes, caption="ภาพที่เลือก", width="stretch")

        direct_plate = st.checkbox(
            "ภาพนี้เป็นป้ายทะเบียนระยะใกล้",
            key="resident_direct_plate",
            help="ใช้กับภาพที่มีป้ายทะเบียนเป็นส่วนหลัก เช่น test4.png ระบบจะอ่านภาพเต็มโดยไม่ต้องหากรอบป้ายก่อน",
        )

        if st.button(
            "อ่านทะเบียนจากภาพ",
            type="secondary",
            key="read_resident_plate",
        ):
            try:
                with st.spinner("กำลังโหลดโมเดลและอ่านทะเบียนจากภาพ กรุณารอสักครู่..."):
                    model = get_yolo_model(MODEL_PATH)
                    pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    annotated, detected_text, confidence, candidates = detect_plate_text(
                        model,
                        frame,
                        conf_threshold=0.10,
                        imgsz=1024,
                        direct_plate=direct_plate,
                    )

                # เตรียมค่าเริ่มต้นของฟอร์ม โดยไม่ลบข้อมูลลูกบ้านช่องอื่น
                st.session_state["new_owner"] = st.session_state.get("new_owner", "")
                st.session_state["new_plate"] = detected_text or ""
                st.session_state["new_house"] = st.session_state.get("new_house", "")
                st.session_state["new_phone"] = st.session_state.get("new_phone", "")
                st.session_state["new_note"] = st.session_state.get("new_note", "")
                st.session_state["new_province"] = province_from_candidates(candidates) or "ไม่ระบุ"

                suggestions = [
                    item for item in candidates
                    if "province" in item and not item.get("province_confirmed", True)
                ]
                st.session_state["resident_province_suggestion"] = (
                    max(suggestions, key=lambda item: item["confidence"])
                    if suggestions else None
                )

                st.session_state["resident_ocr_text"] = detected_text or ""
                st.session_state["resident_ocr_conf"] = float(confidence or 0)
                st.session_state["resident_ocr_preview"] = cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB
                )
                st.session_state["resident_ocr_message"] = (
                    "อ่านทะเบียนสำเร็จ" if detected_text
                    else (
                        "อ่านภาพป้ายโดยตรงแล้ว แต่ไม่พบตัวอักษรที่ผ่านตัวกรอง กรุณาตรวจผลโมเดล"
                        if direct_plate else
                        "ไม่พบผลอ่านทะเบียน หากเป็นภาพป้ายระยะใกล้ ให้เลือกช่องภาพป้ายทะเบียนระยะใกล้แล้วอ่านอีกครั้ง"
                    )
                )
                st.rerun()

            except Exception as exc:
                st.error(f"อ่านภาพไม่สำเร็จ: {exc}")
                st.exception(exc)

    if st.session_state.get("resident_ocr_text"):
        st.success(
            f"ผลอ่านเบื้องต้น: {st.session_state['resident_ocr_text']} "
            f"(Confidence {st.session_state.get('resident_ocr_conf', 0):.2f}) — กรุณาตรวจสอบก่อนบันทึก"
        )
        if st.session_state.get("resident_ocr_preview") is not None:
            st.image(
                st.session_state["resident_ocr_preview"],
                caption="ผลตรวจจับจากโมเดล",
                width="stretch",
            )
    elif st.session_state.get("resident_ocr_message"):
        st.warning(st.session_state["resident_ocr_message"])

    suggestion = st.session_state.get("resident_province_suggestion")
    if suggestion:
        st.warning(
            f"จังหวัดที่โมเดลคาดไว้: {suggestion['province']} "
            f"(คะแนน {suggestion['confidence']:.2f}) — ยังไม่ยืนยัน "
            "กรุณาตรวจดูป้ายแล้วเลือกจังหวัดในฟอร์มก่อนบันทึก"
        )

    provinces = [
        "ไม่ระบุ", "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
        "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร",
        "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม",
        "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
        "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี",
        "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา", "พิจิตร", "พิษณุโลก", "เพชรบุรี",
        "เพชรบูรณ์", "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน", "ยโสธร",
        "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน",
        "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ", "สมุทรสงคราม",
        "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี",
        "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ",
        "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี", "อุบลราชธานี"
    ]
    if st.session_state.get("new_province", "ไม่ระบุ") not in provinces:
        st.session_state["new_province"] = "ไม่ระบุ"

    with st.form("add_resident_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        owner = c1.text_input("ชื่อเจ้าของรถ *", key="new_owner")
        plate = c2.text_input("ทะเบียนรถ *", placeholder="เช่น 9กน 7038", key="new_plate")
        province = st.selectbox(
            "จังหวัดบนป้ายทะเบียน", provinces,
            index=provinces.index(st.session_state.get("new_province", "ไม่ระบุ")),
            key="new_province"
        )
        c3, c4 = st.columns(2)
        house = c3.text_input("บ้านเลขที่ / เฟส", key="new_house")
        phone = c4.text_input("เบอร์โทรศัพท์ (ไม่บังคับ)", key="new_phone")
        note = st.text_input("หมายเหตุ", key="new_note")
        active = st.checkbox("อนุญาตให้เข้าหมู่บ้าน", value=True, key="new_active")
        submitted = st.form_submit_button("เพิ่มทะเบียน", type="primary")
        if submitted:
            try:
                db.add_resident(owner, plate, house, phone, note, active, province=province)
                st.success("เพิ่มข้อมูลเรียบร้อย")
                for key in ("resident_ocr_text", "resident_ocr_conf", "resident_ocr_preview", "resident_province_suggestion"):
                    st.session_state.pop(key, None)
                st.rerun()
            except Exception as exc:
                if "UNIQUE constraint failed" in str(exc):
                    st.error("ทะเบียนนี้มีอยู่ในระบบแล้ว")
                else:
                    st.error(str(exc))

    st.divider()
    st.subheader("ค้นหาและจัดการข้อมูล")
    search = st.text_input("ค้นหาจากชื่อ ทะเบียน จังหวัด บ้านเลขที่ หรือเบอร์โทร")
    rows = db.list_residents(search=search, include_inactive=True)
    if not rows:
        st.info("ยังไม่มีข้อมูลที่ตรงกับการค้นหา")
        return

    st.dataframe(resident_frame(rows), width="stretch", hide_index=True)
    choices = {f'{r["id"]} | {r["plate_text"]} | {r["owner_name"]}': r["id"] for r in rows}
    selected_label = st.selectbox("เลือกข้อมูลที่ต้องการแก้ไขหรือลบ", list(choices.keys()))
    selected_id = choices[selected_label]
    current = db.get_resident(selected_id)
    if current is None:
        st.warning("ไม่พบข้อมูลนี้แล้ว กรุณาโหลดหน้าใหม่")
        return

    with st.form("edit_resident_form"):
        st.markdown("**แก้ไขข้อมูล**")
        c1, c2 = st.columns(2)
        edit_owner = c1.text_input("ชื่อเจ้าของรถ", value=current["owner_name"])
        edit_plate = c2.text_input("ทะเบียนรถ", value=current["plate_text"])
        edit_province = st.selectbox(
            "จังหวัดบนป้ายทะเบียน", provinces,
            index=provinces.index(current["province"] if current["province"] in provinces else "ไม่ระบุ"),
            key=f"edit_province_{selected_id}"
        )
        c3, c4 = st.columns(2)
        edit_house = c3.text_input("บ้านเลขที่ / เฟส", value=current["house_no"] or "")
        edit_phone = c4.text_input("เบอร์โทรศัพท์", value=current["phone"] or "")
        edit_note = st.text_input("หมายเหตุ", value=current["note"] or "")
        edit_active = st.checkbox("อนุญาตให้เข้าหมู่บ้าน", value=bool(current["active"]))
        save = st.form_submit_button("บันทึกการแก้ไข")
        if save:
            try:
                db.update_resident(
                    selected_id, edit_owner, edit_plate, edit_house,
                    edit_phone, edit_note, edit_active, province=edit_province
                )
                st.success("บันทึกข้อมูลแล้ว")
                st.rerun()
            except Exception as exc:
                if "UNIQUE constraint failed" in str(exc):
                    st.error("ทะเบียนนี้มีอยู่ในระบบแล้ว")
                else:
                    st.error(str(exc))

    if st.button("ลบข้อมูลลูกบ้านที่เลือก", type="secondary"):
        st.session_state["confirm_delete_resident_id"] = selected_id
    if st.session_state.get("confirm_delete_resident_id") == selected_id:
        st.warning(f"ยืนยันการลบทะเบียน {current['plate_text']} ของ {current['owner_name']} หรือไม่?")
        yes, no = st.columns(2)
        if yes.button("ยืนยันลบ", type="primary"):
            db.delete_resident(selected_id)
            st.session_state.pop("confirm_delete_resident_id", None)
            st.success("ลบข้อมูลแล้ว")
            st.rerun()
        if no.button("ยกเลิก"):
            st.session_state.pop("confirm_delete_resident_id", None)
            st.rerun()


def process_uploaded_video(model):

          upload = st.file_uploader(

                    "เลือกวิดีโอรถเข้าออก",

                    type=["mp4", "avi", "mov", "mkv"],

                    help="แนะนำ MP4 เพื่อความเข้ากันได้กับ OpenCV"

          )

          conf = st.slider("Confidence ขั้นต่ำ", 0.01, 0.90, 0.10, 0.01, key="upload_conf")

          imgsz = st.select_slider("ขนาดภาพสำหรับตรวจจับ", options=[512, 640, 768, 896, 1024], value=1024)

          frame_step = st.number_input("ตรวจทุกกี่เฟรม (1 = ทุกเฟรม)", min_value=1, max_value=30, value=1)

          if upload is None:

                    st.info("อัปโหลดวิดีโอเพื่อเริ่มตรวจจับ")

                    return



          if st.button("เริ่มตรวจสอบวิดีโอ", type="primary"):

                    suffix = Path(upload.name).suffix or ".mp4"

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:

                              temp.write(upload.getbuffer())

                              temp_path = temp.name

                    cap = cv2.VideoCapture(temp_path)

                    if not cap.isOpened():

                              cap.release()

                              os.unlink(temp_path)

                              st.error("เปิดไฟล์วิดีโอไม่ได้ ลองแปลงเป็น MP4 แล้วอัปโหลดใหม่")

                              return



                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if not fps or fps <= 0:

                              fps = 25.0

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

                    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

                    writer = cv2.VideoWriter(

                              out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,

                              (width, height)

                    )

                    if not writer.isOpened():

                              cap.release()

                              os.unlink(temp_path)

                              st.error("สร้างวิดีโอผลลัพธ์ไม่ได้ ลองใช้ไฟล์ MP4")

                              return



                    progress = st.progress(0)

                    status = st.empty()

                    preview = st.empty()

                    detected_rows = []

                    frame_idx = 0

                    logged = {}

                    try:

                              while True:

                                        ok, frame = cap.read()

                                        if not ok:

                                                  break

                                        if frame_idx % int(frame_step) == 0:

                                                  annotated, plate_text, confidence, candidates = detect_plate_text(

                                                            model, frame, conf_threshold=conf, imgsz=imgsz

                                                  )

                                                  if plate_text:

                                                            detected_province = province_from_candidates(candidates)

                                                            resident, authorized, gate_status = check_gate(plate_text, candidates)

                                                            gate_reason = (
                                                                      "ทะเบียนและจังหวัดตรงกัน" if authorized
                                                                      else ("ไม่พบจังหวัดจากภาพ" if not detected_province else "ไม่พบข้อมูลลูกบ้านที่ตรงทั้งทะเบียนและจังหวัด หรือจังหวัดในฐานข้อมูลยังเป็น 'ไม่ระบุ'")
                                                            )

                                                            cv2.putText(

                                                                      annotated,

                                                                      f"GATE: {'OPEN (SIMULATED)' if authorized else 'CLOSED'}",

                                                                      (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,

                                                                      (0, 180, 0) if authorized else (0, 0, 255), 2

                                                            )

                                                            detected_rows.append({

                                                                      "frame": frame_idx,

                                                                      "time_seconds": round(frame_idx / fps, 2),

                                                                      "plate_text": plate_text,

                                                                      "province": detected_province or "ตรวจไม่พบจังหวัด",

                                                                      "confidence": round(confidence, 3),

                                                                      "authorized": "อนุญาต" if authorized else "ไม่อนุญาต",

                                                                      "gate_status": gate_status,

                                                                      "reason": gate_reason,

                                                                      "owner_name": resident["owner_name"] if resident else ""

                                                            })

                                                            # Record the first result and allow one upgrade from denied to authorized.

                                                            normalized = normalize_for_match(plate_text)

                                                            if normalized and (normalized not in logged or (authorized and not logged[normalized])):

                                                                      db.add_vehicle_log(

                                                                                plate_text, resident, authorized, gate_status,

                                                                                "อัปโหลดวิดีโอ", confidence,

                                                                                f"ไฟล์ {upload.name}, วินาที {frame_idx / fps:.2f}, จังหวัดที่ตรวจพบ: {detected_province or 'ไม่พบ'}, {gate_reason}"

                                                                      )

                                                                      logged[normalized] = bool(authorized)

                                                  else:

                                                            annotated = frame

                                        else:

                                                  annotated = frame



                                        writer.write(annotated)

                                        frame_idx += 1

                                        if total_frames > 0:

                                                  progress.progress(min(frame_idx / total_frames, 1.0))

                                        if frame_idx % max(1, int(frame_step) * 5) == 0:

                                                  status.write(f"กำลังประมวลผล {frame_idx:,} / {total_frames or 'ไม่ทราบ'} เฟรม")

                                                  small = cv2.resize(annotated, (min(800, width), int(height * min(800, width) / width)))

                                                  preview.image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), channels="RGB")



                              cap.release()

                              writer.release()

                              progress.progress(1.0)

                              status.success(f"ประมวลผลเสร็จแล้ว จำนวน {frame_idx:,} เฟรม")

                              if detected_rows:

                                        st.subheader("ผลตรวจจับ")

                                        st.dataframe(pd.DataFrame(detected_rows), width="stretch", hide_index=True)

                              else:

                                        st.warning("ไม่พบป้ายทะเบียนที่ผ่าน Confidence ขั้นต่ำในวิดีโอนี้")

                              with open(out_path, "rb") as f:

                                        st.download_button(

                                                  "ดาวน์โหลดวิดีโอที่ใส่กรอบตรวจจับแล้ว",

                                                  data=f.read(),

                                                  file_name="annotated_vehicle_video.mp4",

                                                  mime="video/mp4"

                                        )

                    finally:

                              cap.release()

                              writer.release()

                              for path in (temp_path,):

                                        try:

                                                  os.unlink(path)

                                        except OSError:

                                                  pass





def live_camera(model):

          st.write("กล้องสดทำงานผ่านกล้องของเครื่องที่เปิดเว็บแอปอยู่ หากเปิดแอปจากเครื่องอื่น กล้องที่ใช้จะเป็นกล้องของเครื่องนั้น")

          st.warning("การต่อกล้องสดต้องติดตั้ง streamlit-webrtc และอนุญาตสิทธิ์กล้องในเบราว์เซอร์")

          try:

                    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

                    import av

          except ImportError:

                    st.error("ยังไม่ได้ติดตั้ง streamlit-webrtc หรือ av กรุณารัน: python -m pip install streamlit-webrtc av")

                    return



          conf = st.slider("Confidence ขั้นต่ำสำหรับกล้องสด", 0.10, 0.90, 0.35, 0.05, key="live_conf")

          imgsz = st.select_slider("ขนาดภาพกล้องสด", options=[512, 640, 768, 896, 1024], value=640, key="live_imgsz")



          class LiveProcessor(VideoProcessorBase):

                    def __init__(self):

                              self.model = model

                              self.conf = conf

                              self.imgsz = imgsz

                              self.last_plate = ""

                              self.last_conf = 0.0

                              self.last_owner = ""

                              self.last_authorized = False

                              self.last_gate = "ปิด"

                              self.last_logged_at = 0.0

                              self.last_log_key = ""

                              self.lock = __import__("threading").Lock()



                    def recv(self, frame):

                              image = frame.to_ndarray(format="bgr24")

                              try:

                                        annotated, plate_text, confidence, candidates = detect_plate_text(

                                                  self.model, image, conf_threshold=self.conf, imgsz=self.imgsz

                                        )

                                        resident, authorized, gate_status = check_gate(plate_text, candidates) if plate_text else (None, False, "ปิด")

                                        with self.lock:

                                                  self.last_plate = plate_text

                                                  self.last_conf = confidence

                                                  self.last_owner = resident["owner_name"] if resident else ""

                                                  self.last_authorized = authorized

                                                  self.last_gate = gate_status

                                                  normalized = normalize_for_match(plate_text)

                                                  now = time.time()

                                                  # Log only when the recognized plate changes or after a 30-second cooldown.

                                                  if normalized and (normalized != self.last_log_key or now - self.last_logged_at >= 30):

                                                            db.add_vehicle_log(

                                                                      plate_text, resident, authorized, gate_status,

                                                                      "กล้องสด", confidence, "ตรวจจับจากกล้องสด"

                                                            )

                                                            self.last_log_key = normalized

                                                            self.last_logged_at = now



                                        label = "GATE: OPEN (SIMULATED)" if authorized else "GATE: CLOSED"

                                        cv2.putText(

                                                  annotated, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,

                                                  (0, 180, 0) if authorized else (0, 0, 255), 2

                                        )

                                        if plate_text:

                                                  cv2.putText(

                                                            annotated, f"{plate_text[:70]} ({confidence:.2f})", (12, 60),

                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2

                                                  )

                                        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

                              except Exception:

                                        return frame



          ctx = webrtc_streamer(

                    key="vehicle-live-camera",

                    video_processor_factory=LiveProcessor,

                    rtc_configuration=RTCConfiguration(

                              {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

                    ),

                    media_stream_constraints={"video": True, "audio": False},

                    async_processing=True,

          )



          if ctx.state.playing and ctx.video_processor:

                    processor = ctx.video_processor



                    @st.fragment(run_every="1s")

                    def live_status():

                              with processor.lock:

                                        plate = processor.last_plate

                                        confidence = processor.last_conf

                                        owner = processor.last_owner

                                        authorized = processor.last_authorized

                                        gate = processor.last_gate

                              c1, c2, c3 = st.columns(3)

                              c1.metric("สถานะประตูจำลอง", gate)

                              c2.metric("Confidence", f"{confidence:.2f}")

                              c3.metric("ผลตรวจสอบ", "อนุญาต" if authorized else "ไม่อนุญาต")

                              st.write("ทะเบียนที่ตรวจพบ:", plate or "ยังไม่พบ")

                              st.write("เจ้าของรถ:", owner or "ไม่พบข้อมูลลูกบ้าน")



                    live_status()





def logs_page():

          st.subheader("ประวัติรถเข้าออก")

          c1, c2 = st.columns([3, 1])

          search = c1.text_input("ค้นหาประวัติจากทะเบียน ชื่อ บ้านเลขที่ หรือแหล่งวิดีโอ")

          limit = c2.selectbox("จำนวนรายการ", [100, 500, 1000, 5000], index=1)

          rows = db.list_vehicle_logs(search=search, limit=limit)

          if rows:

                    frame = pd.DataFrame([dict(row) for row in rows])

                    frame = frame.rename(columns={

                              "id": "ลำดับ", "event_time": "วันเวลา", "plate_text": "ทะเบียนที่ตรวจพบ",

                              "owner_name": "เจ้าของรถ", "house_no": "บ้านเลขที่", "authorized": "อนุญาต",

                              "gate_status": "สถานะประตูจำลอง", "source": "แหล่งตรวจจับ",

                              "confidence": "Confidence", "detail": "รายละเอียด"

                    })

                    if "อนุญาต" in frame:

                              frame["อนุญาต"] = frame["อนุญาต"].map({1: "ใช่", 0: "ไม่ใช่"})

                    st.dataframe(frame, width="stretch", hide_index=True)

                    st.download_button(

                              "ดาวน์โหลดประวัติเป็น CSV",

                              data=frame.to_csv(index=False).encode("utf-8-sig"),

                              file_name="vehicle_logs.csv",

                              mime="text/csv"

                    )

          else:

                    st.info("ยังไม่มีประวัติรถเข้าออก")

          st.divider()

          if st.button("ล้างประวัติทั้งหมด", type="secondary"):

                    st.session_state["confirm_clear_logs"] = True

          if st.session_state.get("confirm_clear_logs"):

                    st.warning("ต้องการลบประวัติรถเข้าออกทั้งหมดหรือไม่? ข้อมูลนี้กู้คืนไม่ได้")

                    yes, no = st.columns(2)

                    if yes.button("ยืนยันล้างประวัติ", type="primary"):

                              db.clear_vehicle_logs()

                              st.session_state.pop("confirm_clear_logs", None)

                              st.success("ล้างประวัติแล้ว")

                              st.rerun()

                    if no.button("ยกเลิก"):

                              st.session_state.pop("confirm_clear_logs", None)

                              st.rerun()





def main():

          st.title("ระบบจัดการรถเข้าออกหมู่บ้าน")

          st.caption("YOLO License Plate Detection + SQLite + Streamlit")

          with st.sidebar:

                    st.header("เมนู")

                    page = st.radio("เลือกหน้า", [

                              "ภาพรวม", "จัดการทะเบียนลูกบ้าน", "ตรวจสอบวิดีโอ", "กล้องสด", "ประวัติรถเข้าออก"

                    ])

                    st.divider()

                    st.write("ไฟล์โมเดล")

                    st.code(MODEL_PATH, language="text")

                    st.caption("หากขึ้นว่าไม่พบ best.pt ให้แก้ MODEL_PATH ใน app.py")



          if page == "ภาพรวม":

                    show_dashboard()

          elif page == "จัดการทะเบียนลูกบ้าน":

                    resident_management()

          elif page == "ตรวจสอบวิดีโอ":

                    st.subheader("ตรวจสอบวิดีโอที่อัปโหลด")

                    try:

                              model = get_yolo_model(MODEL_PATH)

                              process_uploaded_video(model)

                    except Exception as exc:

                              st.error(f"โหลดโมเดลไม่สำเร็จ: {exc}")

                              st.info("ตรวจสอบว่าไฟล์ best.pt อยู่ในตำแหน่งที่ตั้งไว้ และติดตั้ง ultralytics แล้ว")

          elif page == "กล้องสด":

                    st.subheader("ตรวจจับป้ายทะเบียนจากกล้องสด")

                    try:

                              model = get_yolo_model(MODEL_PATH)

                              live_camera(model)

                    except Exception as exc:

                              st.error(f"โหลดโมเดลหรือเริ่มกล้องไม่สำเร็จ: {exc}")

          elif page == "ประวัติรถเข้าออก":

                    logs_page()





if __name__ == "__main__":

          main()
