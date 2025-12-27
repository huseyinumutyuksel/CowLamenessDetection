
import zipfile
import xml.etree.ElementTree as ET
import os

def read_docx(file_path):
    try:
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # XML namespace for Word
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text = []
            for node in tree.iter():
                if node.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t':
                    if node.text:
                        text.append(node.text)
                elif node.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p':
                     text.append('\n')
            
            return "".join(text)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    file_path = r"c:/Users/Umut/PycharmProjects/CowLamenessDetection/talimatlar/Inek Topallık Seviye Tespiti ve Segmantasyonu.docx"
    print(read_docx(file_path))
