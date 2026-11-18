"""Patch: Fix admin nav and expand it in index.html"""

with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

old = '''                <!-- Admin Nav Menu -->
                <div class="glass-panel hidden" id="adminNav">
                    <h2><i class="fa-solid fa-chart-pie"></i> Platform Metrics</h2>
                    <ul class="nav-menu">
                        <li class="nav-item active" data-tab="analytics" onclick="switchTab('admin', 'analytics')">
                            <i class="fa-solid fa-chart-line"></i> Dashboard KPIs
                        </li>
                    </ul>
                </div>'''

new = '''                <!-- Admin Nav Menu -->
                <div class="glass-panel hidden" id="adminNav">
                    <h2><i class="fa-solid fa-shield-halved" style="color:var(--primary);"></i> <span data-ar="\u0644\u0648\u062d\u0629 \u0627\u0644\u0625\u062f\u0627\u0631\u0629" data-en="Admin Panel">Admin Panel</span></h2>
                    <ul class="nav-menu">
                        <li class="nav-item active" data-tab="analytics" onclick="switchTab('admin', 'analytics')">
                            <i class="fa-solid fa-chart-line" style="color:#2563eb;"></i>
                            <span data-ar="\u0644\u0648\u062d\u0629 \u0627\u0644\u0625\u062d\u0635\u0627\u0626\u064a\u0627\u062a" data-en="Analytics Dashboard">Analytics Dashboard</span>
                        </li>
                        <li class="nav-item" onclick="loadAdminUsers()">
                            <i class="fa-solid fa-users" style="color:#7c3aed;"></i>
                            <span data-ar="\u0625\u062f\u0627\u0631\u0629 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645\u064a\u0646" data-en="User Management">User Management</span>
                        </li>
                        <li class="nav-item" onclick="loadAdminPharmacies()">
                            <i class="fa-solid fa-store" style="color:#059669;"></i>
                            <span data-ar="\u0625\u062f\u0627\u0631\u0629 \u0627\u0644\u0635\u064a\u062f\u0644\u064a\u0627\u062a" data-en="Pharmacies">Pharmacies</span>
                        </li>
                    </ul>
                    <hr style="margin:12px 0; border-color:var(--border);">
                    <p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-muted);padding:0 4px;margin-bottom:6px;">
                        <span data-ar="\u0623\u062f\u0648\u0627\u062a \u0627\u0644\u0646\u0638\u0627\u0645" data-en="System Tools">System Tools</span>
                    </p>
                    <ul class="nav-menu">
                        <li class="nav-item" onclick="showPanel(\'tab-dosage-calc\')">
                            <i class="fa-solid fa-calculator" style="color:#2563eb;"></i>
                            <span data-ar="\u062d\u0627\u0633\u0628\u0629 \u0627\u0644\u062c\u0631\u0639\u0629" data-en="Dosage Calculator">Dosage Calculator</span>
                        </li>
                        <li class="nav-item" onclick="showPanel(\'tab-cdss\')">
                            <i class="fa-solid fa-stethoscope" style="color:#c084fc;"></i>
                            <span data-ar="\u062f\u0639\u0645 \u0627\u0644\u0642\u0631\u0627\u0631 \u0627\u0644\u0633\u0631\u064a\u0631\u064a" data-en="Clinical Decision">Clinical Decision</span>
                        </li>
                        <li class="nav-item" onclick="window.open(\'/drug-graph\',\'_blank\')">
                            <i class="fa-solid fa-circle-nodes" style="color:#f59e0b;"></i>
                            <span data-ar="\u062a\u0641\u0627\u0639\u0644\u0627\u062a \u0627\u0644\u0623\u062f\u0648\u064a\u0629" data-en="Drug Interactions">Drug Interactions</span>
                        </li>
                    </ul>
                </div>'''

if old in content:
    content = content.replace(old, new)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Admin nav expanded!")
else:
    print("ERROR: Could not find target text")
    # Debug
    idx = content.find('adminNav')
    print(f"adminNav found at index: {idx}")
    print(repr(content[idx-50:idx+200]))
