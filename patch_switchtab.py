"""Fix switchTab and any remaining display issues in index.html"""

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix switchTab - replace the problematic lines
old_switch = "            document.querySelectorAll('.dashboard-view').forEach(view => {\r\n                view.classList.remove('active');\r\n                view.style.display = '';\r\n            });\r\n            \r\n            const viewId = `${role}-${tab}`;\r\n            const el = document.getElementById(viewId);\r\n            if (el) el.classList.add('active');"

new_switch = """            // Hide all panels
            document.querySelectorAll('.dashboard-view').forEach(view => {
                view.classList.remove('active');
                view.style.display = 'none';
            });
            // Show target
            const viewId = `${role}-${tab}`;
            const el = document.getElementById(viewId);
            if (el) { el.classList.add('active'); el.style.display = 'block'; }"""

if old_switch in content:
    content = content.replace(old_switch, new_switch)
    print("switchTab fixed!")
else:
    # Try with \n instead of \r\n
    old_switch2 = old_switch.replace('\r\n', '\n')
    if old_switch2 in content:
        content = content.replace(old_switch2, new_switch)
        print("switchTab fixed (LF)!")
    else:
        print("WARNING: switchTab old text not found, trying partial match...")
        # Fix just the display='' line
        content = content.replace("view.style.display = '';", "view.style.display = 'none';")
        # Fix the classList.add line for switchTab
        old_el = "if (el) el.classList.add('active');"
        new_el = "if (el) { el.classList.add('active'); el.style.display = 'block'; }"
        count = content.count(old_el)
        print(f"Found '{old_el}' {count} times")
        content = content.replace(old_el, new_el)
        print("Partial fix applied!")

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! File saved.")
