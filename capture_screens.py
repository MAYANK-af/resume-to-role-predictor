"""
capture_screens.py
Automates browser screenshots using Selenium + Edge for:
1. Landing state (simplified, headline, textarea, dropdown, analyze button)
2. Analyzing state (3D words flying)
3. Result state (predicted role, fit score, top 5 words, See why)
"""

import time
import os
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

os.makedirs("screenshots", exist_ok=True)

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1920,1080")
options.add_argument("--enable-webgl")
options.add_argument("--hide-scrollbars")

print("Launching Edge driver...")
driver = webdriver.Edge(options=options)
driver.set_window_size(1920, 1080)

try:
    print("Navigating to http://127.0.0.1:3000...")
    driver.get("http://127.0.0.1:3000")
    time.sleep(3.5)

    # 1. Screenshot Landing State
    landing_path = os.path.abspath("screenshots/landing.png")
    driver.save_screenshot(landing_path)
    print(f"Captured: {landing_path}")

    # Find and click a sample button (e.g. 'AI / ML')
    sample_buttons = driver.find_elements(By.TAG_NAME, "button")
    for btn in sample_buttons:
        if "AI / ML" in btn.text:
            print(f"Clicking sample button: {btn.text}...")
            driver.execute_script("arguments[0].click();", btn)
            break
    time.sleep(1.2)

    # 2. Click Analyze and capture Analyzing State
    analyze_btn = None
    for btn in driver.find_elements(By.TAG_NAME, "button"):
        if "analyze" in btn.text.lower():
            analyze_btn = btn
            break

    if analyze_btn:
        print(f"Clicking '{analyze_btn.text}' button...")
        driver.execute_script("arguments[0].click();", analyze_btn)
        time.sleep(0.35)  # In flight
        analyzing_path = os.path.abspath("screenshots/analyzing.png")
        driver.save_screenshot(analyzing_path)
        print(f"Captured: {analyzing_path}")

    # 3. Wait for results to render and capture Result State
    time.sleep(2.0)
    driver.execute_script("""
        const el = document.getElementById('results-view');
        if (el) {
            el.scrollIntoView({ behavior: 'instant', block: 'start' });
            window.scrollBy(0, -40);
        }
    """)
    time.sleep(0.8)

    result_path = os.path.abspath("screenshots/result.png")
    driver.save_screenshot(result_path)
    print(f"Captured: {result_path}")

    print("All screenshots captured successfully!")

finally:
    driver.quit()
