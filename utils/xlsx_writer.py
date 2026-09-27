"""Small dependency-free XLSX writer for tabular experiment outputs."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def _column_name(index: int) -> str:
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _value(value):
    if value is None:
        return ""
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
    except TypeError:
        pass
    return str(value)


def write_xlsx(path: str | Path, sheets: dict[str, object]) -> None:
    """Write DataFrame-like objects with ``columns`` and ``itertuples``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    rels = []
    workbook = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>',
    ]
    for index, (name, frame) in enumerate(sheets.items(), start=1):
        safe_name = escape(str(name)[:31])
        workbook.append(f'<sheet name="{safe_name}" sheetId="{index}" r:id="rId{index}"/>')
        rels.append(f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>')
        content_types.append(f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    workbook.append('</sheets></workbook>')
    content_types.append('</Types>')
    workbook_xml = ''.join(workbook)
    workbook_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(rels) + '</Relationships>'
    root_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', ''.join(content_types))
        archive.writestr('_rels/.rels', root_rels)
        archive.writestr('xl/workbook.xml', workbook_xml)
        archive.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
        for index, (_, frame) in enumerate(sheets.items(), start=1):
            columns = list(frame.columns)
            rows = [columns] + [list(row) for row in frame.itertuples(index=False, name=None)]
            xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>']
            for row_index, row in enumerate(rows, start=1):
                xml.append(f'<row r="{row_index}">')
                for col_index, value in enumerate(row):
                    ref = f'{_column_name(col_index)}{row_index}'
                    text = escape(_value(value))
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                        xml.append(f'<c r="{ref}" t="n"><v>{text}</v></c>')
                    else:
                        xml.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
                xml.append('</row>')
            xml.append('</sheetData></worksheet>')
            archive.writestr(f'xl/worksheets/sheet{index}.xml', ''.join(xml))
