import os
from nicegui import ui, app, Client
from typing import Dict, List, Optional
import asyncio
import logging

from state import StateManager

import logging
import sys

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout
)
os.makedirs('projects_data', exist_ok=True)
app.add_static_files('/projects_data', 'projects_data')

@ui.page('/')
async def main_page(client: Client):
    logging.info("New client connected. Setting up main page.")
    state = StateManager()
    ui.dark_mode().enable()
    ui.add_body_html('<style>body { overflow: hidden; }</style>')

    paper_ui_elements: Dict[str, Dict] = {}
    article_chat_history: List[Dict[str, str]] = []
    chat_container_ref: Dict[str, Optional[ui.column]] = {'value': None}
    content_container_ref: Dict[str, Optional[ui.html]] = {'value': None}
    scroll_area_ref: Dict[str, Optional[ui.scroll_area]] = {'value': None}

    with ui.left_drawer(value=False).classes('bg-gray-100 dark:bg-gray-800') as drawer:
        with ui.column().classes('w-full p-4'):
            ui.label("Проекты").classes('text-h6')
            with ui.row().classes('w-full items-center'):
                new_project_name = ui.input("Название нового проекта").props('dense').classes('flex-grow')
                ui.button(icon='add', on_click=lambda: state.create_project(new_project_name.value), color='positive').props('flat')
            ui.separator()
            project_list_container = ui.column().classes('w-full')

    with ui.column().classes('w-screen h-screen p-4 gap-4'):
        paper_list_view = ui.column().classes('w-full h-full flex flex-col no-wrap')
        article_workspace_view = ui.column().classes('w-full h-full flex flex-col no-wrap').style('display: none;')

    def switch_view(view_name: str):
        paper_list_view.style(f'display: {"flex" if view_name == "list" else "none"}')
        article_workspace_view.style(f'display: {"flex" if view_name == "workspace" else "none"}')

    
    async def do_notify(message: str, type: str):
        ui.notify(message, type=type)

    async def update_project_list_ui():
        project_list_container.clear()
        with project_list_container:
            for proj in state.get_projects():
                is_current = state.current_project_id == proj['id']
                bg_color = 'bg-gray-200 dark:bg-gray-800' if is_current else ''
                async def handle_select(p_id): await state.select_project(p_id); drawer.hide()
                with ui.item(on_click=lambda p_id=proj['id']: handle_select(p_id)).classes(f'w-full cursor-pointer rounded px-2 {bg_color}'):
                    with ui.item_section(): ui.label(proj['name'])
                    ui.space()
                    ui.button(icon='delete', on_click=lambda e, p_id=proj['id']: state.delete_project(p_id), color='negative').props('flat round dense').on('click.stop')

    async def build_paper_list_view(project_data: Dict | None):
        nonlocal paper_ui_elements
        paper_ui_elements.clear()
        paper_list_view.clear()
        with paper_list_view:
            with ui.row().classes('w-full items-center p-2 shrink-0'):
                ui.button(icon='menu', on_click=drawer.toggle).props('flat round')
                ui.label(f"Проект: {project_data['name']}" if project_data else "Проект не выбран").classes('text-h6 ml-4')
            with ui.row().classes('w-full items-center px-2'):
                search_query = ui.input(placeholder="Найти статьи на arXiv...").classes('flex-grow')
                search_button = ui.button("Поиск", on_click=lambda: state.search_and_add_papers(search_query.value))
                if not project_data: search_button.disable()
            ui.separator().classes('shrink-0 my-2')

            with ui.scroll_area().classes('w-full flex-grow p-2'):
                if not project_data:
                    with ui.column().classes('w-full h-full justify-center items-center'):
                        ui.label("Выберите или создайте проект").classes('text-2xl text-gray-400')
                    return

                papers = project_data.get("papers", [])
                if not papers:
                    ui.label("Статьи еще не добавлены.").classes('m-auto')
                else:
                    for paper in papers:
                        with ui.card().classes('w-full mb-2'):
                            with ui.row().classes('w-full items-center no-wrap'):
                                with ui.column().classes('flex-grow'):
                                    ui.label(paper['title']).classes('font-bold')
                                    
                                    status = paper.get("status", "new")
                                    color = 'positive' if status == 'done' else ('negative' if status == 'error' else 'grey')
                                    badge = ui.badge(status, color=color)

                                with ui.row().classes('shrink-0 ml-4'):
                                    open_button = ui.button(icon='open_in_new', on_click=lambda p=paper: build_article_workspace(p)) \
                                        .props('flat color=primary').tooltip('Открыть')
                                    
                                    if status != 'done':
                                        open_button.disable()
                                        
                                    ui.button(icon='delete', on_click=lambda p=paper: state.delete_paper(p['id']), color='negative') \
                                        .props('flat').tooltip('Удалить')
                        
                        paper_ui_elements[paper['id']] = {'badge': badge, 'button': open_button}
                                        
        switch_view('list')
    
    def build_article_workspace(paper_stub: Dict):
        nonlocal article_chat_history
        paper = state.get_paper_details(paper_stub['id'])
        if not paper:
            state._run_ui_update(do_notify("Не удалось загрузить актуальные данные статьи.", "negative"))
            return

        article_chat_history.clear()
        article_workspace_view.clear()
        
        def update_content_display(paper_data: Dict, view_type: str):
            container = content_container_ref.get('value')
            if not container: return

            if view_type == 'html':
                # paper_data['html_path'] хранит что-то вроде 'content/1603.08029v1/index.html'
                html_file_path = paper_data.get('html_path')
                if html_file_path:
                    iframe_src = f"/projects_data/{html_file_path}"
                    container.set_content(f'<iframe src="{iframe_src}" class="w-full h-full border-0"></iframe>')
                else:
                    container.set_content('<p>HTML путь для этой статьи не найден.</p>')

            elif view_type == 'pdf':
                pdf_url = paper_data.get('url', '').replace('http://', 'https://')
                if pdf_url:
                    container.set_content(f'<iframe src="https://docs.google.com/gview?url={pdf_url}&embedded=true" class="w-full h-full border-0"></iframe>')
                else:
                    container.set_content('<p>PDF URL для этой статьи не найден.</p>')

        with article_workspace_view:
            with ui.splitter(value=60).classes('w-full h-full') as splitter:
                with splitter.before:
                    with ui.column().classes('w-full h-full no-wrap'):
                        with ui.row().classes('w-full items-center p-2 bg-slate-100 dark:bg-slate-800 shrink-0'):
                            ui.button("Назад к списку", on_click=lambda: switch_view('list')).props('flat')
                            ui.space()
                            ui.toggle(
                                {'html': 'HTML', 'pdf': 'PDF'},
                                value='html',
                                on_change=lambda e: update_content_display(paper, e.value)
                            ).props('dense')

                        content_container = ui.html().classes('w-full flex-grow')
                        content_container_ref['value'] = content_container

                with splitter.after:
                    with ui.column().classes('w-full h-full flex flex-col no-wrap'):
                        ui.label("Анализ статьи").classes('text-h6 p-4 shrink-0')
                        with ui.scroll_area().classes('flex-grow w-full border-t border-b p-2 bg-slate-200 dark:bg-slate-800') as scroll_area:
                            scroll_area_ref['value'] = scroll_area
                            chat_messages_container = ui.column().classes('w-full')
                            chat_container_ref['value'] = chat_messages_container
                        with ui.row().classes('w-full items-center p-2 shrink-0'):
                            text_input = ui.input(placeholder="Задайте вопрос...").classes('flex-grow').props('outlined') \
                                .on('keydown.enter', lambda: (state.start_article_analysis(paper['id'], text_input.value, article_chat_history), text_input.set_value(None)))
        
        update_content_display(paper, 'html')
        asyncio.create_task(update_article_chat_ui([]))
        switch_view('workspace')

    async def update_article_chat_ui(messages: List[Dict]):
        nonlocal article_chat_history
        article_chat_history = messages
        
        chat_container = chat_container_ref.get('value')
        scroll_area = scroll_area_ref.get('value')
        if not chat_container or not scroll_area: 
            return

        chat_container.clear()
        with chat_container:
            for msg in messages:
                with ui.chat_message(name=msg.get('author'), sent=msg.get('author') == "Вы"):
                    ui.markdown(msg.get('text', '').strip()).classes('text-body1 w-full')

        scroll_area.scroll_to(percent=1.0)


    async def update_paper_status_in_ui(paper_id: str, status: str):
        elements = paper_ui_elements.get(paper_id)
        if not elements:
            return

        badge = elements['badge']
        button = elements['button']

        badge.set_text(status)
        if status == 'done':
            badge.props('color=positive')
            button.enable()
        elif status == 'error':
            badge.props('color=negative')
            button.disable()
        else:
            badge.props('color=grey')
            button.disable()

    loop = asyncio.get_running_loop()
    state.register_ui_callbacks(
        loop=loop,
        update_project_list=update_project_list_ui,
        update_paper_view=build_paper_list_view,
        update_article_chat=update_article_chat_ui,
        notify_ui=do_notify,
        update_paper_status=update_paper_status_in_ui
    )
    
    await state.initialize_for_client()

ui.run(title="AI Research Assistant", storage_secret="my_secret_key", native=True, window_size=(1800, 1000), reload=False)