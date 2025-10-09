import os
import json
from typing import List, Dict, Optional
import threading

class ProjectStore:
    def __init__(self, storage_path: str = "projects_data"):
        self.storage_path = storage_path
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        self.locks: Dict[str, threading.RLock] = {}
        self._dict_lock = threading.Lock()

    def _get_lock(self, project_id: str) -> threading.RLock:
        with self._dict_lock:
            if project_id not in self.locks:
                self.locks[project_id] = threading.RLock()
            return self.locks[project_id]

    def _get_project_path(self, project_id: str) -> str:
        return os.path.join(self.storage_path, f"{project_id}.json")

    def _read_project_file(self, project_path: str) -> Optional[Dict]:
        if not os.path.exists(project_path):
            return None
        try:
            with open(project_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"STORE WARNING: Пропущен или не удалось прочитать поврежденный файл: {os.path.basename(project_path)}")
            return None

    def create_project(self, name: str) -> Dict:
        project_id = name.lower().replace(" ", "_").strip().replace("-", "_")
        lock = self._get_lock(project_id)
        with lock:
            project_path = self._get_project_path(project_id)
            project_data = self._read_project_file(project_path)
            if project_data:
                return project_data

            new_project_data = {"id": project_id, "name": name, "papers": []}
            with open(project_path, 'w', encoding='utf-8') as f:
                json.dump(new_project_data, f, ensure_ascii=False, indent=4)
            return new_project_data

    def get_projects(self) -> List[Dict]:
        projects = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                project_data = self._read_project_file(os.path.join(self.storage_path, filename))
                if project_data:
                    projects.append(project_data)
        return projects

    def get_project(self, project_id: str) -> Optional[Dict]:
        lock = self._get_lock(project_id)
        with lock:
            return self._read_project_file(self._get_project_path(project_id))

    def _save_project_unsafe(self, project_data: Dict):
        project_path = self._get_project_path(project_data['id'])
        with open(project_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False, indent=4)

    def _modify_project(self, project_id: str, modification_func):
        lock = self._get_lock(project_id)
        with lock:
            project = self._read_project_file(self._get_project_path(project_id))
            if project:
                modification_func(project)
                self._save_project_unsafe(project)
    
    def add_paper_to_project(self, project_id: str, paper_data: Dict):
        def add_paper(project):
            if not any(p.get('id') == paper_data.get('id') for p in project.get('papers', [])):
                project.setdefault('papers', []).append(paper_data)
        self._modify_project(project_id, add_paper)

    def delete_paper_from_project(self, project_id: str, paper_id: str):
        """НОВАЯ ФУНКЦИЯ: Удаляет статью из проекта по ее ID."""
        def remove_paper(project):
            papers = project.get('papers', [])
            project['papers'] = [p for p in papers if p.get('id') != paper_id]
        self._modify_project(project_id, remove_paper)

    def update_paper_status(self, project_id: str, paper_id: str, status: str):
        def update_status(project):
            for paper in project.get('papers', []):
                if paper.get('id') == paper_id:
                    paper['status'] = status
                    break
        self._modify_project(project_id, update_status)

    def save_processed_paper(self, project_id: str, paper_id: str, html_path: str):
        def save_path(project):
            for paper in project.get('papers', []):
                if paper.get('id') == paper_id:
                    paper['html_path'] = html_path
                    break
        self._modify_project(project_id, save_path)

    def delete_project(self, project_id: str) -> bool:
        lock = self._get_lock(project_id)
        with lock:
            project_path = self._get_project_path(project_id)
            if os.path.exists(project_path):
                try:
                    os.remove(project_path)
                    print(f"STORE: Проект {project_id} удален.")
                    with self._dict_lock:
                        if project_id in self.locks:
                            del self.locks[project_id]
                    return True
                except OSError as e:
                    print(f"STORE ERROR: Не удалось удалить файл проекта {project_id}: {e}")
                    return False
            return False