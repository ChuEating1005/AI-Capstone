from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import os
import requests
from PIL import Image
from io import BytesIO
import random

# 下載圖片函式
def download_image(url, folder, index, downloaded_hashes=None):
    if downloaded_hashes is None:
        downloaded_hashes = set()
        
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 檢查請求是否成功
        
        # 計算圖片內容的哈希值，用於檢測重複
        content_hash = hash(response.content)
        if content_hash in downloaded_hashes:
            print(f"跳過重複圖片: {url}")
            return False
            
        downloaded_hashes.add(content_hash)
        
        image = Image.open(BytesIO(response.content))
        image.save(os.path.join(folder, f"{index}.jpg"))
        print(f"下載成功: {index}.jpg")
        return True  # 返回成功標誌
    except Exception as e:
        print(f"下載失敗: {url}, 原因: {e}")
        return False  # 返回失敗標誌

def search_google_images(query):
    # 啟動 Selenium 瀏覽器
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 隱藏瀏覽器視窗
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)

    # 進入 Google Images
    google_url = f"https://www.google.com/search?tbm=isch&q={query}"
    driver.get(google_url)
    time.sleep(3)  # 增加等待時間

    # 模擬滾動 (載入更多圖片)
    print("尋找 Google 圖片中...")
    for _ in range(3):  # 滾動 5 次
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        print(f"滾動中... {_ + 1}")
        time.sleep(3)  # 增加等待時間

    # 獲取所有圖片網址 - 更新選擇器
    image_elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i, img.YQ4gaf")  # 更新的Google圖片選擇器
    image_urls = []
    for img in image_elements:
        src = img.get_attribute("src")
        if src:
            image_urls.append(src)
        else:
            # 有時圖片網址存在data-src屬性中
            data_src = img.get_attribute("data-src")
            if data_src:
                image_urls.append(data_src)

    # 關閉瀏覽器
    driver.quit()

    print(f"找到 {len(image_urls)} 張圖片網址")
    return image_urls

def search_unsplash(query):
    # 啟動 Selenium 瀏覽器
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 隱藏瀏覽器視窗
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)

    unsplash_url = f"https://unsplash.com/s/photos/{query}"
    driver.get(unsplash_url)
    time.sleep(3)  # 增加等待時間

    # 模擬滾動 (載入更多圖片)
    print("尋找 Unsplash 圖片中...")
    for _ in range(3):  # 滾動 5 次
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        print(f"滾動中... {_ + 1}")
        time.sleep(3)  # 增加等待時間

    # 嘗試多種可能的選擇器
    selectors = [
        "img.tB6UZ",                  # 原始選擇器
        "img.YVj9w",                  # 可能的新選擇器
        "figure img",                 # 通用選擇器
        ".MorZF img",                 # 另一個可能的選擇器
        "div[data-test='search-photos-grid'] img"  # 基於測試屬性的選擇器
    ]
    
    image_urls = []
    for selector in selectors:
        print(f"嘗試選擇器: {selector}")
        image_elements = driver.find_elements(By.CSS_SELECTOR, selector)
        if image_elements:
            print(f"選擇器 {selector} 找到 {len(image_elements)} 個元素")
            for img in image_elements:
                # 嘗試獲取各種可能的屬性
                for attr in ["src", "data-src"]:
                    url = img.get_attribute(attr)
                    if url and url.startswith("http") and url not in image_urls:
                        image_urls.append(url)
                
                # 檢查srcset屬性
                srcset = img.get_attribute("srcset")
                if srcset:
                    # 從srcset中提取最大圖片URL
                    urls = srcset.split(',')
                    if urls:
                        largest_url = urls[-1].strip().split(' ')[0]
                        if largest_url.startswith("http") and largest_url not in image_urls:
                            image_urls.append(largest_url)

    # 如果還是沒找到圖片，嘗試直接執行JavaScript獲取所有圖片
    if not image_urls:
        print("嘗試使用JavaScript獲取所有圖片...")
        image_urls = driver.execute_script("""
            const images = Array.from(document.querySelectorAll('img'));
            return images
                .map(img => img.src || img.dataset.src)
                .filter(src => src && src.startsWith('http'));
        """)

    # 關閉瀏覽器
    driver.quit()

    print(f"找到 {len(image_urls)} 張 Unsplash 圖片網址")
    return image_urls

# 設定關鍵字 & 儲存目錄
search_queries = ["T-shirt", "Jacket", "Pants", "Shoes", "Bag", "Hat", "Glasses", "Dress", "Hoodie", "Socks"]
for query in search_queries:
    save_directory = f"images/{query}"
    os.makedirs(save_directory, exist_ok=True)

    google_urls = search_google_images(query)
    unsplash_urls = search_unsplash(query)

    # 合併兩個列表並隨機排序
    all_urls = google_urls[:60] + unsplash_urls[:60]
    random.shuffle(all_urls)

    # 下載圖片
    index = 0
    downloaded_hashes = set()  # 用於存儲已下載圖片的哈希值
    for url in all_urls:
        if download_image(url, save_directory, index, downloaded_hashes):
            index += 1  # 只有在下載成功時才增加索引

    print(f"完成 {query} 的圖片下載")
