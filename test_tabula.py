import tabula

pdf_path = r"D:\Rag\1-s2.0-S1877050926007295-main.pdf"

tables = tabula.read_pdf(
    pdf_path,
    pages=6,
    multiple_tables=True,
    stream=True
)

print("Tables found:", len(tables))

for i, table in enumerate(tables, start=1):
    print(f"\nTABLE {i}")
    print(table.to_string(index=False))