def clean_text(text,allowed_chars = set("""ก, ข, ค, ฆ, ง, จ, ฉ, ช, ซ, ฌ, ญ, ฎ, ฏ, ฐ, ฑ, ฒ, ณ, ด, ต, ถ, ท, ธ, น, บ, ป, ผ, ฝ, พ, ฟ, ภ, ม, ย, ร, ล, ว, ศ, ษ, ส, ห, ฬ, อ, ฮ, ฤ, ฦ, ฯ,  
ะ, ั, า, ำ, ิ, ี, ึ, ื, ุ, ู, เ, แ, โ, ใ, ไ,  
ๆ, ็, ่, ้, ๊, ๋, ์,  
๑, ๒, ๓, ๔, ๕, ๖, ๗, ๘, ๙, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  
?
a b c d e f g h i j k l m n o p q r s t u v w x y z
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
""")):
    
    return "".join(c for c in text if c in allowed_chars)

if __name__ == "__main__":
    input_text = "สวัสดี ,"
    cleaned_text = clean_text(input_text)