"""Fix showPanel and switchTab in index.html to use only class-based display"""

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

count = 0

# Fix showPanel
old1 = """        function showPanel(panelId) {
            // Hide all dashboard views
            document.querySelectorAll('.dashboard-view').forEach(v => {
                v.classList.remove('active');
                v.style.display = 'none';
            });
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            // Show the target panel
            const panel = document.getElementById(panelId);
            if (panel) {
                panel.classList.add('active');
                panel.style.display = 'block';
                // Scroll to top
                panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }"""

new1 = """        function showPanel(panelId) {
            // Hide all dashboard views using class only (CSS handles display)
            document.querySelectorAll('.dashboard-view').forEach(v => {
                v.classList.remove('active');
                v.removeAttribute('style');
            });
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            // Show the target panel
            const panel = document.getElementById(panelId);
            if (panel) {
                panel.classList.add('active');
            }
        }"""

if old1 in content:
    content = content.replace(old1, new1)
    count += 1
    print("showPanel fixed!")
else:
    print("WARNING: showPanel not found with exact match")

# Fix switchTab - replace only the display logic part
old2 = "            // Hide all panels\n            document.querySelectorAll('.dashboard-view').forEach(view => {\n                view.classList.remove('active');\n                view.style.display = 'none';\n            });\n            // Show target\n            const viewId = `${role}-${tab}`;\n            const el = document.getElementById(viewId);\n            if (el) { el.classList.add('active'); el.style.display = 'block'; }"

new2 = """            // Hide all panels (CSS .active class handles display)
            document.querySelectorAll('.dashboard-view').forEach(view => {
                view.classList.remove('active');
                view.removeAttribute('style');
            });
            // Show target
            const viewId = `${role}-${tab}`;
            const el = document.getElementById(viewId);
            if (el) { el.classList.add('active'); }"""

if old2 in content:
    content = content.replace(old2, new2)
    count += 1
    print("switchTab fixed!")
else:
    # Try LF version
    old2_lf = old2.replace('\r\n', '\n')
    if old2_lf in content:
        content = content.replace(old2_lf, new2)
        count += 1
        print("switchTab fixed (LF)!")
    else:
        print("WARNING: switchTab display block not found")
        # Remove all inline style sets from these functions
        content = content.replace("view.style.display = 'none';", "view.removeAttribute('style');")
        content = content.replace("el.style.display = 'block';", "")
        content = content.replace("panel.style.display = 'block';", "")
        print("Applied fallback fix")
        count += 1

with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done! Applied {count} fixes.")
