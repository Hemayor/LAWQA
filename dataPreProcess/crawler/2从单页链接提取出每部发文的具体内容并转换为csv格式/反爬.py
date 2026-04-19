# -*- coding: utf-8 -*-
import os
import re
import csv
import time
import random
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import requests

# -----------------------------
# 配置
# -----------------------------
TXT_PATH = "../test.txt"
OUTPUT_CSV = "pkulaw_legalv1.csv"
FAILED_URLS_FILE = "../failed_urls.txt"
LIST_PAGE_URL = "https://www.pkulaw.com/"
BATCH_SIZE = 10  # 每批爬取数量，可自己调节
SCROLL_TIMES = 2  # 每条正文页滚动次数

# UA池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/117.0.5938.92 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0",
]

# -----------------------------
# 工具函数
# -----------------------------
def random_ua():
    return random.choice(USER_AGENTS)

def sleep_random(min_sec=1, max_sec=3):
    time.sleep(random.uniform(min_sec, max_sec))

def write_failed_line(line_content):
    with open(FAILED_URLS_FILE, "a", encoding="utf-8") as f:
        f.write(line_content + "\n")

# -----------------------------
# 初始化文件
# -----------------------------
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)
    print(f"⚠️ 已删除旧文件: {OUTPUT_CSV}")

if os.path.exists(FAILED_URLS_FILE):
    os.remove(FAILED_URLS_FILE)
    print(f"⚠️ 已删除旧文件: {FAILED_URLS_FILE}")

csvfile = open(OUTPUT_CSV, "w", encoding="utf-8", newline="")
writer = csv.DictWriter(csvfile, fieldnames=[
    "title", "law_code", "promulgating_authority", "published_date",
    "effective_date", "legal_level", "category", "chapters", "url"
])
writer.writeheader()

# -----------------------------
# 读取链接并打乱顺序
# -----------------------------
urls_with_line = []
with open(TXT_PATH, "r", encoding="utf-8") as f:
    for line in f.readlines():
        if line.strip():  # 非空行
            urls_with_line.append(line.strip())

random.shuffle(urls_with_line)
print(f"📄 共发现 {len(urls_with_line)} 个法律链接，已打乱顺序")

urls_with_title = []
for line in urls_with_line:
    parts = line.split("\t")
    if len(parts) == 2:
        title_text, url = parts
        urls_with_title.append((title_text, url, line))

# -----------------------------
# 抓取函数
# -----------------------------
def extract_law_content(page, url):
    headers = {"User-Agent": random_ua()}
    try:
        # requests快速检查可访问性
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            raise Exception(f"Requests返回状态码 {resp.status_code}")

        # Playwright访问正文页
        page.set_extra_http_headers({
            "User-Agent": random_ua(),
            "Referer": LIST_PAGE_URL,
            "Accept-Language": "zh-CN,zh;q=0.9",
        })
        page.goto(url)
        page.wait_for_load_state("networkidle")
        sleep_random(1, 3)

        # 模拟滚动阅读
        for _ in range(SCROLL_TIMES):
            page.evaluate("window.scrollBy(0, window.innerHeight);")
            sleep_random(1, 3)

        html = page.content()
        soup = BeautifulSoup(html, "html.parser")

        # -----------------------------
        # 基本信息
        # -----------------------------
        title_tag = soup.find("h2", class_="title")
        title = title_tag.find(string=True, recursive=False).strip() if title_tag else ""

        law_code_tag = soup.find("a", string=re.compile(r"CLI\."))
        law_code = law_code_tag.get_text(strip=True) if law_code_tag else ""

        # # 如果没有law_code，认为抓取失败
        # if not law_code:
        #     raise Exception("未获取到 law_code")

        promulgating_authority_tag = soup.find("a", attrs={"logfunc": "制定机关"})
        promulgating_authority = promulgating_authority_tag.get_text(strip=True) if promulgating_authority_tag else ""

        def get_date(label):
            tag = soup.find("strong", string=re.compile(label))
            if tag:
                return tag.parent.get_text(strip=True).replace(label, "").replace(".", "-")
            return ""

        published_date = get_date("公布日期：")
        effective_date = get_date("施行日期：")

        legal_level_tag = soup.find("a", attrs={"logfunc": "效力位阶"})
        legal_level = legal_level_tag.get_text(strip=True) if legal_level_tag else ""

        category_tags = soup.select("div.box span a[logfunc='法规类别']")
        category = [t.get_text(strip=True) for t in category_tags]

        # -----------------------------
        # 章节 + 条文
        # -----------------------------
        chapters = []

        chap_tags = soup.select("p.navzhang")
        if chap_tags:
            for chap in chap_tags:
                chapter_title = chap.get_text(strip=True).replace("　", " ")
                chapter = {"chapter_title": chapter_title, "articles": []}

                for sib in chap.find_all_next():
                    if sib.name == "p" and "navzhang" in sib.get("class", []):
                        break
                    if sib.name == "div" and "tiao-wrap" in sib.get("class", []):
                        tiao_tag = sib.select_one("span.navtiao")
                        article_number = tiao_tag.get_text(strip=True) if tiao_tag else ""
                        kuan_contents = sib.select("div.kuan-content")
                        content = "\n".join(
                            kc.get_text(strip=True).replace(article_number, "").strip() for kc in kuan_contents)
                        judicial_tag = sib.select_one("a[href^='/clink/pfnl']")
                        judicial_case = "https://www.pkulaw.com" + judicial_tag["href"] if judicial_tag else ""
                        relevant_laws = []
                        for kc in kuan_contents:
                            for a in kc.select("a.alink"):
                                name = a.get_text(strip=True)
                                if name not in relevant_laws:
                                    relevant_laws.append(name)
                        chapter["articles"].append({
                            "article_number": article_number,
                            "content": content,
                            "judicial_case": judicial_case,
                            "relevant_laws": relevant_laws
                        })
                chapters.append(chapter)
        else:
            chapter = {"chapter_title": "第一章", "articles": []}
            for sib in soup.select("div.tiao-wrap"):
                tiao_tag = sib.select_one("span.navtiao")
                article_number = tiao_tag.get_text(strip=True) if tiao_tag else ""
                kuan_contents = sib.select("div.kuan-content")
                content = "\n".join(kc.get_text(strip=True).replace(article_number, "").strip() for kc in kuan_contents)
                judicial_tag = sib.select_one("a[href^='/clink/pfnl']")
                judicial_case = "https://www.pkulaw.com" + judicial_tag["href"] if judicial_tag else ""
                relevant_laws = []
                for kc in kuan_contents:
                    for a in kc.select("a.alink"):
                        name = a.get_text(strip=True)
                        if name not in relevant_laws:
                            relevant_laws.append(name)
                chapter["articles"].append({
                    "article_number": article_number,
                    "content": content,
                    "judicial_case": judicial_case,
                    "relevant_laws": relevant_laws
                })
            chapters.append(chapter)

        # -----------------------------
        # 失败判定：首章无条文
        # -----------------------------
        if not chapters or not chapters[0].get("articles"):
            raise Exception("章节存在但首章 articles 为空，判定为抓取失败")

        return {
            "title": title,
            "law_code": law_code,
            "promulgating_authority": promulgating_authority,
            "published_date": published_date,
            "effective_date": effective_date,
            "legal_level": legal_level,
            "category": category,
            "chapters": chapters,
            "url": url
        }

    except Exception as e:
        raise e

# -----------------------------
# 批量爬取
# -----------------------------
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    for idx, (title_text, url, original_line) in enumerate(urls_with_title, 1):
        print(f"\n➡️ ({idx}/{len(urls_with_title)}) 开始处理")
        print(f"🔍 正在抓取: {title_text} {url}")

        # 每隔随机 2~3 条正文访问列表页一次
        if idx % random.randint(2, 3) == 0:
            try:
                print(f"🔄 访问列表页降低封禁风险: {LIST_PAGE_URL}")
                page.goto(LIST_PAGE_URL)
                page.wait_for_load_state("networkidle")
                sleep_random(2, 4)
            except:
                pass

        try:
            law_data = extract_law_content(page, url)
            writer.writerow(law_data)
        except Exception as e:
            print(f"❌ 抓取失败，已记录到 {FAILED_URLS_FILE} -> {e}")
            write_failed_line(original_line)

        # 自然等待
        sleep_random(2, 4)

        # 长休眠每 20~30 条
        if idx % random.randint(30, 40) == 0:
            long_sleep = random.uniform(10, 20)
            print(f"💤 长休眠 {long_sleep:.1f} 秒")
            time.sleep(long_sleep)

    browser.close()

csvfile.close()
print("\n✅ 全部完成，数据已保存，失败链接已记录到 failed_urls.txt")
