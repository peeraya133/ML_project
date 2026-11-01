import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "residents.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS residents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_name TEXT NOT NULL,
                plate_text TEXT NOT NULL UNIQUE,
                house_no TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                note TEXT DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                province TEXT NOT NULL DEFAULT 'ไม่ระบุ'
            )
        """)
        # Migration for an existing database created before province was added.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(residents)").fetchall()}
        if "province" not in columns:
            conn.execute("ALTER TABLE residents ADD COLUMN province TEXT NOT NULL DEFAULT 'ไม่ระบุ'")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                owner_name TEXT DEFAULT '',
                house_no TEXT DEFAULT '',
                authorized INTEGER NOT NULL,
                gate_status TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 0,
                detail TEXT DEFAULT ''
            )
        """)
        conn.commit()


def normalize_plate(value: str) -> str:
    """Normalize a plate for matching: uppercase and keep letters/digits only."""
    return "".join(ch for ch in str(value).upper().strip() if ch.isalnum())


def list_residents(search: str = "", include_inactive: bool = True):
    sql = "SELECT * FROM residents WHERE 1=1"
    params = []
    if not include_inactive:
        sql += " AND active = 1"
    if search.strip():
        needle = f"%{search.strip()}%"
        sql += " AND (owner_name LIKE ? OR plate_text LIKE ? OR province LIKE ? OR house_no LIKE ? OR phone LIKE ?)"
        params.extend([needle, needle, needle, needle, needle])
    sql += " ORDER BY id DESC"
    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()


def get_resident(resident_id: int):
    with get_connection() as conn:
        return conn.execute("SELECT * FROM residents WHERE id = ?", (resident_id,)).fetchone()


def add_resident(owner_name, plate_text, house_no="", phone="", note="", active=True, province="ไม่ระบุ"):
    owner_name = owner_name.strip()
    plate_text = plate_text.strip().upper()
    province = (province or "ไม่ระบุ").strip()
    if not owner_name:
        raise ValueError("กรุณากรอกชื่อเจ้าของรถ")
    if not normalize_plate(plate_text):
        raise ValueError("กรุณากรอกทะเบียนรถ")
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO residents
            (owner_name, plate_text, house_no, phone, note, active, created_at, province)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (owner_name, plate_text, house_no.strip(), phone.strip(), note.strip(),
              1 if active else 0, datetime.now().isoformat(timespec="seconds"), province))
        conn.commit()


def update_resident(resident_id, owner_name, plate_text, house_no="", phone="", note="", active=True, province="ไม่ระบุ"):
    owner_name = owner_name.strip()
    plate_text = plate_text.strip().upper()
    province = (province or "ไม่ระบุ").strip()
    if not owner_name:
        raise ValueError("กรุณากรอกชื่อเจ้าของรถ")
    if not normalize_plate(plate_text):
        raise ValueError("กรุณากรอกทะเบียนรถ")
    with get_connection() as conn:
        conn.execute("""
            UPDATE residents
            SET owner_name=?, plate_text=?, house_no=?, phone=?, note=?, active=?, province=?
            WHERE id=?
        """, (owner_name, plate_text, house_no.strip(), phone.strip(), note.strip(),
              1 if active else 0, province, resident_id))
        conn.commit()


def delete_resident(resident_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM residents WHERE id = ?", (resident_id,))
        conn.commit()


def normalize_province(value: str) -> str:
    """Normalize province names so 'จังหวัดชลบุรี' matches 'ชลบุรี'."""
    value = str(value or "").strip()
    if value.startswith("จังหวัด"):
        value = value[len("จังหวัด"):].strip()
    return value


def find_authorized_resident(detected_plate: str, detected_province: str | None = None):
    """Return an active resident only when both plate and province match."""
    target_plate = normalize_plate(detected_plate)
    target_province = normalize_province(detected_province)

    if not target_plate or not target_province or target_province == "ไม่ระบุ":
        return None

    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM residents WHERE active = 1").fetchall()

    for row in rows:
        saved_plate = normalize_plate(row["plate_text"])
        saved_province = normalize_province(row["province"])
        if saved_plate == target_plate and saved_province != "ไม่ระบุ" and saved_province == target_province:
            return row
    return None


def add_vehicle_log(plate_text, resident, authorized, gate_status, source, confidence=0.0, detail=""):
    owner = resident["owner_name"] if resident else ""
    house = resident["house_no"] if resident else ""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO vehicle_logs
            (event_time, plate_text, owner_name, house_no, authorized, gate_status, source, confidence, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(timespec="seconds"), plate_text, owner, house,
              1 if authorized else 0, gate_status, source, float(confidence), detail))
        conn.commit()


def list_vehicle_logs(search="", limit=500):
    sql = "SELECT * FROM vehicle_logs"
    params = []
    if search.strip():
        needle = f"%{search.strip()}%"
        sql += " WHERE plate_text LIKE ? OR owner_name LIKE ? OR house_no LIKE ? OR source LIKE ?"
        params.extend([needle, needle, needle, needle])
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()


def clear_vehicle_logs():
    with get_connection() as conn:
        conn.execute("DELETE FROM vehicle_logs")
        conn.commit()
