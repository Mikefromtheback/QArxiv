import threading
import logging
from typing import Callable, Awaitable, Dict, List
import asyncio
import shutil
import os
from nicegui import ui, app

from core.data.project_store import ProjectStore
from core.tools.paper_fetcher import search_papers
from core.tools.paper_analyzer import analyze_paper
from core.ingestion.content_extractor import download_and_parse_html_offline

log = logging.getLogger(__name__)

class StateManager:
    def __init__(self):
        log.debug("StateManager instance created.")
        self.project_store = ProjectStore(storage_path="projects_data")
        self.current_project_id: str | None = None
        self.paper_ui_elements: Dict[str, Dict] = {}
        
        self.loop: asyncio.AbstractEventLoop | None = None
        self.update_project_list: Callable[[], Awaitable[None]] | None = None
        self.update_paper_view: Callable[[Dict | None], Awaitable[None]] | None = None
        self.update_article_chat: Callable[..., Awaitable[None]] | None = None
        self.update_paper_status_ui: Callable[[str, str], Awaitable[None]] | None = None
        self.notify_ui: Callable[[str, str], Awaitable[None]] | None = None

    def register_ui_callbacks(self, *, loop, update_project_list, update_paper_view, update_article_chat, notify_ui, update_paper_status):
        log.debug("UI callbacks registered.")
        self.loop = loop
        self.update_project_list = update_project_list
        self.update_paper_view = update_paper_view
        self.update_article_chat = update_article_chat
        self.update_paper_status_ui = update_paper_status
        self.notify_ui = notify_ui


    def set_paper_ui_elements(self, elements: Dict[str, Dict]):
        """Сохраняет ссылки на UI элементы карточек статей."""
        self.paper_ui_elements = elements

    def _run_ui_update(self, coro: Awaitable):
        if self.loop and coro:
            asyncio.run_coroutine_threadsafe(coro, self.loop)

    def _notify(self, message: str, type: str):
        log.info(f"Scheduling notification: '{message}' with type '{type}'")
        if self.notify_ui and self.loop:
            coro = self.notify_ui(message, type=type)
            self._run_ui_update(coro)

    def get_projects(self):
        return self.project_store.get_projects()

    async def initialize_for_client(self):
        log.info("Initializing state for new client.")
        self.current_project_id = app.storage.user.get('current_project_id')
        await self.select_project(self.current_project_id)

    def create_project(self, name: str):
        if not name: return
        project = self.project_store.create_project(name)
        self._run_ui_update(self.select_project(project['id']))
        self._notify(f"Проект '{name}' создан.", 'positive')

    def delete_project(self, project_id: str):
        if self.project_store.delete_project(project_id):
            if self.current_project_id == project_id:
                self._run_ui_update(self.select_project(None))
            else:
                self._run_ui_update(self.update_project_list())
            self._notify('Проект удален.', 'info')
        else:
            self._notify('Не удалось удалить проект', 'negative')

    async def select_project(self, project_id: str | None):
        self.current_project_id = project_id
        app.storage.user['current_project_id'] = project_id
        await self.update_project_list()
        project_data = self.project_store.get_project(project_id) if project_id else None
        await self.update_paper_view(project_data)

    def search_and_add_papers(self, query: str):
        if not self.current_project_id:
            self._notify("Сначала выберите или создайте проект.", "warning")
            return
        if not query:
            self._notify("Введите поисковый запрос.", "warning")
            return
            
        self._notify(f"Идет поиск по запросу: '{query}'...", 'info')
        thread = threading.Thread(target=self._search_task, args=(self.current_project_id, query), name="SearchThread")
        thread.start()
        
    def _search_task(self, project_id: str, query: str):
        found_papers = search_papers(query)
        if not found_papers:
            self._notify(f"По запросу '{query}' ничего не найдено.", 'info')
            return
        for paper_meta in found_papers:
            self.project_store.add_paper_to_project(project_id, paper_meta)
        project_data = self.project_store.get_project(project_id)
        self._run_ui_update(self.update_paper_view(project_data))
        self._notify(f"Найдено {len(found_papers)} статей. Начинаю загрузку.", 'positive')
        ingestion_thread = threading.Thread(target=self._process_papers_sequentially, args=(project_id, found_papers), name="PaperIngestionThread")
        ingestion_thread.start()
        
    def _process_papers_sequentially(self, project_id, paper_list):
        for paper in paper_list:
            self._ingest_paper_task(project_id, paper['id'])
        self._notify("Загрузка и обработка статей завершена.", 'positive')

    def _ingest_paper_task(self, project_id: str, paper_id: str):
        self.project_store.update_paper_status(project_id, paper_id, "processing")
        if self.update_paper_status_ui:
            self._run_ui_update(self.update_paper_status_ui(paper_id, "processing"))

        try:
            content_dir = os.path.join(self.project_store.storage_path, "content")
            os.makedirs(content_dir, exist_ok=True)
            relative_html_path = download_and_parse_html_offline(paper_id, content_dir)
            self.project_store.save_processed_paper(project_id, paper_id, os.path.join("content", relative_html_path))
            self.project_store.update_paper_status(project_id, paper_id, "done")
        
        except Exception as e:
            log.error(f"Ошибка в _ingest_paper_task для статьи {paper_id}.", exc_info=True)
            self.project_store.update_paper_status(project_id, paper_id, "error")
            self._notify(f"Ошибка обработки '{paper_id}'.", 'negative')
            if self.update_paper_status_ui:
                self._run_ui_update(self.update_paper_status_ui(paper_id, "error"))
        
        else:
            if self.update_paper_status_ui:
                self._run_ui_update(self.update_paper_status_ui(paper_id, "done"))

    def get_paper_details(self, paper_id: str) -> Dict | None:
        if not self.current_project_id:
            return None
        project_data = self.project_store.get_project(self.current_project_id)
        if not project_data:
            return None
        
        for paper in project_data.get("papers", []):
            if paper.get("id") == paper_id:
                return paper
        return None

    def delete_paper(self, paper_id: str):
        if not self.current_project_id: return
        self.project_store.delete_paper_from_project(self.current_project_id, paper_id)
        try:
            content_dir = os.path.join(self.project_store.storage_path, "content", paper_id)
            if os.path.isdir(content_dir): 
                shutil.rmtree(content_dir, ignore_errors=True)
        except Exception as e:
            log.error(f"Could not delete content file for paper {paper_id}: {e}")
        self._notify(f"Статья {paper_id} удалена", 'info')
        self._run_ui_update(self.update_paper_view(self.project_store.get_project(self.current_project_id)))

    def start_article_analysis(self, paper_id: str, question: str, current_chat_history: List[Dict]):
        if not self.current_project_id:
            self._notify("Проект не выбран.", "warning")
            return
        new_history = current_chat_history + [{"author": "Вы", "text": question}]
        self._run_ui_update(self.update_article_chat(new_history))
        thread = threading.Thread(
            target=self._run_analyzer_task, 
            args=(self.current_project_id, paper_id, question, new_history), 
            name="ArticleAnalyzerThread"
        )
        thread.start()

    def _run_analyzer_task(self, project_id: str, paper_id: str, question: str, chat_history: List[Dict]):
        answer_text = analyze_paper(paper_id=paper_id, question=question, project_id=project_id)
        final_history = chat_history + [{"author": "Анализатор", "text": answer_text}]
        
        if self.update_article_chat:
            self._run_ui_update(self.update_article_chat(final_history))
        log.info("Article analysis task finished.")