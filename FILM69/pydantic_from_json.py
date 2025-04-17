import json
from typing import Any, Literal, List, Union, Optional
from collections import OrderedDict
from pydantic import BaseModel, Field, create_model

# namespace สำหรับ eval
_type_ns = {
    'Literal': Literal,
    'List': List,
    'Union': Union,
    'Any': Any,
    'Optional': Optional,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
}

def parse_type(type_str: str):
    """
    แปลง type string ให้เป็น annotation จริง ๆ ด้วย eval
    รองรับ:
     - Literal[...]
     - Union[...]
     - List[...]
     - Any, str, int, float, bool
     - Optional[...] หรือ Optional[..., None]
    """
    # Optional[...] (อาจประกอบด้วย None)
    if type_str.startswith("Optional["):
        inner = type_str[len("Optional["):-1]
        parts = [p.strip() for p in inner.split(",") if p.strip() and p.strip() != "None"]
        if len(parts) != 1:
            raise ValueError(f"Optional ต้องมีแค่ type เดียว (หรือ None): {type_str}")
        base = parse_type(parts[0])
        return Optional[base]

    # Union[...]
    if type_str.startswith("Union["):
        return eval(type_str, {}, _type_ns)

    # Literal[...]
    if type_str.startswith("Literal["):
        return eval(type_str, {}, _type_ns)

    # List[...]
    if type_str.startswith("List["):
        return eval(type_str, {}, _type_ns)

    # พื้นฐาน
    if type_str in _type_ns:
        return _type_ns[type_str]

    raise ValueError(f"Unknown type: {type_str}")

def pydantic_from_json(json_string: str) -> type[BaseModel]:
    raw = json.loads(json_string)
    fields_def: dict[str, tuple[type, Field]] = {}

    # สร้าง fields_def ตามลำดับ JSON
    for field_name, info in raw.items():
        type_str    = info.pop("type")
        title       = info.pop("title", field_name)
        description = info.pop("description", "")
        default     = info.pop("default", ...)  # ... คือ required
        extra       = info if info else {}

        annotation = parse_type(type_str)
        fields_def[field_name] = (
            annotation,
            Field(default, title=title, description=description, **extra)
        )

    Model = create_model("DynamicUser", **fields_def)

    return Model



if __name__ == "__main__":
    json_input = '''
{
    "prefix": {
        "type": "Literal['นาย', 'นาง', 'นางสาว']",
        "title": "คำนำหน้าชื่อ",
        "description": "คำนำหน้าชื่อ เช่น นาย, นาง, นางสาว"
    },
    "name": {
        "type": "str",
        "title": "ชื่อ",
        "description": "ชื่อ"
    },
    "surname": {
        "type": "str",
        "title": "นามสกุล",
        "description": "นามสกุล"
    },
    "id": {
        "type": "int",
        "title": "เลขบัตรประชาชน/เลขประจำตัวคนพิการ",
        "description": "เลขบัตรประชาชนหรือเลขประจำตัวคนพิการ",
        "json_schema_extra": {"lenght": 13}
    },
    "disability": {
        "type": "Literal['การเห็น', 'การได้ยิน', 'การเคลื่อนไหว', 'จิตใจ', 'สติปัญญา', 'การเรียนรู้', 'ออทิสติก']",
        "title": "ประเภทความพิการ",
        "description": "ประเภทความพิการเช่น การเห็น การได้ยิน การเคลื่อนไหว หรืออื่นๆ"
    },
    "status": {
        "type": "Literal['โสด', 'สมรส', 'อื่น ๆ']",
        "title": "สถานภาพ",
        "description": "สถานภาพ โสด สมรส หรืออื่น ๆ"
    },
    "education": {
        "type": "Literal['ต่ำกว่าประถมศึกษา', 'ประถมศึกษา', 'มัธยมศึกษาตอนต้น', 'มัธยมศึกษาตอนปลายหรือเทียบเท่า', 'ประกาศนียบัตรวิชาชีพ', 'ประกาศนียบัตรวิชาชีพชั้นสูง', 'ปริญญาตรี', 'ปริญญาโท', 'ปริญญาเอก', 'อื่น ๆ']",
        "title": "การศึกษา",
        "description": "ระดับการศึกษาสูงสุดเช่น ปริญญาตรี, ป.โท ,ป เอก หรืออื่นๆ ใช้ตัวย่อได้"
    },
    "zip_code": {
        "type": "int",
        "title": "รหัสไปรษณีย์",
        "description": "กรุณาระบุรหัสไปรษณีย์ หากไม่ทราบ ให้ระบุชื่อจังหวัดแทน",
        "json_schema_extra": {"lenght": 5}
    },
    "province": {
        "type": "str",
        "title": "จังหวัด",
        "description": "จังหวัดและผู้ใช้ตอบมาแล้วให้ตรวจสอบในฐานข้อมูลจังหวัด"
    },
    "district": {
        "type": "str",
        "title": "อำเภอ/เขต",
        "description": "อำเภอ/เขตและผู้ใช้ตอบมาแล้วให้ตรวจสอบในฐานข้อมูลจังหวัด"
    },
    "subdistrict": {
        "type": "str",
        "title": "แขวง/ตำบล",
        "description": "แขวง/ตำบลและผู้ใช้ตอบมาแล้วให้ตรวจสอบในฐานข้อมูลจังหวัด"
    },
    "email": {
        "type": "Optional[str,None]",
        "title": "อีเมล",
        "description": "อีเมล"
    },
    "types_mor_35": {
        "type": "Literal['สัมปทาน', 'สถานที่จาหน่ายสินค้าหรือบริการ', 'จ้างเหมาจ้างเหมาบริการโดยวิธีกรณีพิเศษ', 'ฝึกงาน', 'จัดให้มีอุปกรณ์หรือสิ่งอานวยความสะดวก', 'ล่ามภาษามือ', 'ให้ความช่วยเหลืออื่นใด']",
        "title": "ประเภทการขอใช้สิทธิตามมาตรา 35",
        "description": "ประเภทการขอใช้สิทธิตามมาตรา 35"
    },
    "status": {
        "type": "Union[Literal['active', 'inactive'], str]",
        "title": "สถานะ",
        "description": "สถานะของผู้ใช้"
    }
}
'''
    UserModel = pydantic_from_json(json_input)
    print(UserModel.model_fields)
