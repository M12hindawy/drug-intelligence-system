import re
with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()
matches = re.findall(r'data-ar="([^"]+)"', content)[:8]
print("data-ar samples:", matches)
arabic_count = sum(1 for c in content if '\u0600' <= c <= '\u06ff')
print("Arabic chars in file:", arabic_count)
print("File OK:", arabic_count > 50)
