import re
import json
import logging 
import time    

log = logging.getLogger(__name__)

class Agent:
    def __init__(self, llm_service, tools: list):
        self.llm = llm_service
        self.tools = {tool.__name__: tool for tool in tools}
        self.system_prompt = """
Ты работаешь по методологии ReAct.
ПРАВИЛА РАБОТЫ:
1.  **Рассуждай (Thought):** На каждом шаге объясняй свою логику.
2.  **Действуй (Action):** Используй один из доступных инструментов для сбора информации.
3.  **Анализируй (Observation):** После получения ответа от инструмента, задай себе главный вопрос: **"Выполнил ли я все что просил пользователь?"**
    -   Если **НЕТ**, продолжай цикл: снова рассуждай и вызывай следующий необходимый инструмент.
    -   Если **ДА**, твой следующий и ПОСЛЕДНИЙ шаг — это предоставить `final_answer`.


ДОСТУПНЫЕ ИНСТРУМЕНТЫ (TOOLS):
{tools_description}

СТРОЖАЙШЕЕ ПРАВИЛО ФОРМАТА ВЫВОДА:
Твой ответ ВСЕГДА должен быть **ТОЛЬКО** одним валидным JSON-объектом.
**ЗАПРЕЩЕНО** добавлять любой текст, комментарии, объяснения или теги до или после JSON-объекта.

ПРИМЕР ВЫЗОВА ИНСТРУМЕНТА:
```json
{{
  "thought": "Мне нужно найти статьи. Я использую search_papers.",
  "action": {{
    "tool_name": "search_papers",
    "arguments": {{
      "query": "tensor train"
    }}
  }}
}}```
ПРИМЕР ФИНАЛЬНОГО ОТВЕТА:
```json
{{
  "thought": "Я сделал все что требовалось с помощью инструментов и теперь готов предоставить final_answer.",
  "final_answer": "Я нашел требуемые статьи и готов их обсудить."
}}```
"""
    def run(self, task: str, project_id: str = None):
        tools_description = "\n".join([f"- {name}: {func.__doc__}" for name, func in self.tools.items()])
        prompt_history = self.system_prompt.format(tools_description=tools_description)
        prompt_history += f"\n\nТекущая задача: {task}"

        for i in range(10): 
            log.info(f"--- АГЕНТ: Шаг {i+1} ---")
            full_prompt = prompt_history
            
            log.info("[АГЕНТ DEBUG] Вызов LLM для получения action...")
            llm_output = self.llm.invoke(full_prompt)
            log.info("[АГЕНТ DEBUG] Ответ от LLM получен.")

            json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', llm_output, re.DOTALL)
            if not json_match:
                yield {"type": "error", "content": f"LLM вернула невалидный ответ без JSON на шаге {i+1}: {llm_output}"}
                return

            response_json_str = next(g for g in json_match.groups() if g)

            try:
                log.info("[АГЕНТ DEBUG] Парсинг JSON...")
                response_data = json.loads(response_json_str)
                log.info("[АГЕНТ DEBUG] JSON успешно распарсен.")
                
                thought_text = response_data.get("thought", "(Нет мысли)")
                yield {"type": "thought", "content": thought_text}

                prompt_history += f"\n\nThought: {thought_text}"
                prompt_history += f"\nAction: {json.dumps(response_data.get('action'), ensure_ascii=False, indent=2)}"


                if "action" in response_data:
                    action_data = response_data["action"]
                    tool_name = action_data.get("tool_name")
                    tool_args = action_data.get("arguments", {})

                    yield {"type": "action", "content": json.dumps(action_data, ensure_ascii=False)}

                    if tool_name and tool_name in self.tools:
                        if 'project_id' in self.tools[tool_name].__code__.co_varnames and 'project_id' not in tool_args:
                            tool_args['project_id'] = project_id
                        
                        try:
                            log.info(f"[АГЕНТ DEBUG] Выполнение инструмента: {tool_name}...")
                            start_time = time.time()
                            observation = self.tools[tool_name](**tool_args)
                            log.info(f"[АГЕНТ DEBUG] Инструмент {tool_name} выполнен за {time.time() - start_time:.2f} сек.")

                            log.info("[АГЕНТ DEBUG] Преобразование observation в текст...")
                            start_time = time.time()
                            if isinstance(observation, (list, dict)):
                                observation_text_for_llm = json.dumps(observation, ensure_ascii=False, indent=2)
                            else:
                                observation_text_for_llm = str(observation)
                            log.info(f"[АГЕНТ DEBUG] Преобразование завершено за {time.time() - start_time:.2f} сек.")

                            yield {"type": "observation", "content": observation}
                            
                            log.info("[АГЕНТ DEBUG] Добавление observation в историю...")
                            prompt_history += f"\nObservation: {observation_text_for_llm}"
                            log.info("[АГЕНТ DEBUG] Observation добавлен.")

                        except Exception as e:
                            error_msg = f"Ошибка при выполнении '{tool_name}': {e}"
                            log.error(f"[АГЕНТ КРИНЖ] {error_msg}", exc_info=True)
                            yield {"type": "error", "content": error_msg}
                            prompt_history += f"\nObservation: Error: {error_msg}"

                    else:
                        error_msg = f"Инструмент '{tool_name}' не найден."
                        yield {"type": "error", "content": error_msg}
                        prompt_history += f"\nObservation: Error: {error_msg}"

                elif "final_answer" in response_data:
                    answer_text = response_data["final_answer"]
                    yield {"type": "final_answer", "content": answer_text}
                    return
                else:
                    yield {"type": "error", "content": "Ответ LLM не содержит ни 'action', ни 'final_answer'."}
                    return

            except json.JSONDecodeError as e:
                error_msg = f"Не удалось распарсить JSON от LLM: {e}. Ответ был: {response_json_str}"
                yield {"type": "error", "content": error_msg}
                return
            except Exception as e:
                log.error(f"[АГЕНТ КРИНЖ] Непредвиденная ошибка в цикле агента.", exc_info=True)
                yield {"type": "error", "content": f"Непредвиденная ошибка в цикле агента: {e}"}
                return

        yield {"type": "error", "content": "Агент не смог завершить задачу за 10 шагов."}