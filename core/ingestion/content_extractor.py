import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
import uuid

log = logging.getLogger(__name__)

def download_resource(url, session):
    try:
        response = session.get(url, timeout=20, stream=True)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        log.warning(f"Не удалось скачать ресурс {url}: {e}")
        return None

def download_and_parse_html_offline(arxiv_id: str, content_storage_path: str) -> str:
    base_url = f"https://ar5iv.org/html/{arxiv_id}"
    session = requests.Session()
    
    try:
        response = session.get(base_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        log.error(f"Не удалось скачать главную страницу статьи {arxiv_id}: {e}")
        error_html = f"<html><body><h1>Ошибка</h1><p>Не удалось скачать содержимое статьи {arxiv_id}.</p><p>Детали: {e}</p></body></html>"
        error_dir = os.path.join(content_storage_path, arxiv_id)
        os.makedirs(error_dir, exist_ok=True)
        error_file_path = os.path.join(error_dir, "index.html")
        with open(error_file_path, "w", encoding="utf-8") as f:
            f.write(error_html)
        return os.path.join(arxiv_id, "index.html")

    article_dir = os.path.join(content_storage_path, arxiv_id)
    assets_dir = os.path.join(article_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    resources_to_download = {}

    for tag in soup.find_all('link', href=True):
        if 'stylesheet' in tag.get('rel', []):
            url = urljoin(base_url + '/', tag['href'])
            filename = os.path.basename(urlparse(url).path) or f"style_{uuid.uuid4().hex}.css"
            local_path = os.path.join(assets_dir, filename)
            resources_to_download[url] = local_path
            tag['href'] = os.path.join("assets", filename) 
            
    for tag_name in ['img', 'script']:
        for tag in soup.find_all(tag_name, src=True):
            url = urljoin(base_url + '/', tag['src'])
            filename = os.path.basename(urlparse(url).path) or f"{tag_name}_{uuid.uuid4().hex}"
            local_path = os.path.join(assets_dir, filename)
            resources_to_download[url] = local_path
            tag['src'] = os.path.join("assets", filename)

    log.info(f"Найдено {len(resources_to_download)} ресурсов для скачивания для статьи {arxiv_id}.")
    for url, local_path in resources_to_download.items():
        content = download_resource(url, session)
        if content:
            with open(local_path, 'wb') as f:
                f.write(content)

    style_tag = soup.new_tag('style')
    style_tag.string = """
        main, .ltx_page_content, article { max-width: none !important; }
        .ltx_figure_panel { margin-left: 0 !important; }
        @media (prefers-color-scheme: dark) {
            body { background-color: #212121 !important; color: #E0E0E0 !important; }
            .ltx_page_content, .ltx_document, .ltx_page, .ltx_authors, .ltx_abstract, 
            .ltx_bibliography, .ltx_biblist li, .ltx_theorem, .ltx_proof,
            .ltx_para, .ltx_p, .ltx_title, .ltx_note, .ltx_section, .ltx_subsection {
                background: #212121 !important; color: #E0E0E0 !important;
            }
            a { color: #90CAF9; } .ltx_cite a, .ltx_ref_tag { color: #90CAF9 !important; }
            .ltx_tag_equation { color: #CE93D8; }
        }
    """
    soup.head.append(style_tag)
    
    for nav_element in soup.select('nav, .ltx_page_logo'):
        nav_element.decompose()
            
    main_html_path = os.path.join(article_dir, "index.html")
    with open(main_html_path, "w", encoding='utf-8') as f:
        f.write(str(soup))
        
    log.info(f"Статья {arxiv_id} и все ресурсы успешно сохранены локально.")
    
    return os.path.join(arxiv_id, "index.html")