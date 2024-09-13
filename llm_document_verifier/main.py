import os
import json
import pandas as pd
import gradio as gr
import logging
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_core.runnables import RunnableSequence

config: dict[str, any] = {}
logger: logging.Logger = None
langchain: RunnableSequence = None


def load_config() -> dict[str, any]:
    """Load configuration from YAML file."""
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def setup_logging() -> None:
    """Set up logging configuration."""
    global logger
    log_level: str = config["logging"]["level"]
    logging.basicConfig(level=log_level, format='%(message).100s')
    logger = logging.getLogger(__name__)


def initialize_llm() -> None:
    """Initialize the LLM with the provided configuration."""
    global langchain
    load_dotenv(override=True)
    api_key: str = os.getenv('OPENAI_API_KEY')

    llm = ChatOpenAI(model=config["openai"]["model"],
                     openai_api_key=api_key,
                     temperature=config["openai"]["temperature"])

    prompt_template: str = """
    System: {system_prompt}
    User: {user_prompt}
    """
    prompt = PromptTemplate(input_variables=["system_prompt", "user_prompt"],
                            template=prompt_template)

    # Create a RunnableSequence
    langchain = prompt | llm


def clean_json_prefix_suffix(text: str) -> str:
    """Clean JSON text of extra characters or code blocks."""
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.endswith('```'):
        text = text[:-3]
    return text


def run_llm_to_get_json_from_contract(contract_file: gr.File,
                                      output_file_path: str) -> str:
    """Extract text from a DOCX file and retry if JSON loading fails."""
    contract2json_system_prompt: str = config["prompts"][
        "contract2json_system"]

    contract_text: str = Docx2txtLoader(
        contract_file.name).load()[0].page_content
    response_cleaned: str = ""

    for attempt in range(config["openai"]["max_retries"]):
        response = langchain.invoke(
            input={
                "system_prompt": contract2json_system_prompt,
                "user_prompt": contract_text
            })

        # Access content using .content attribute
        response_cleaned = clean_json_prefix_suffix(response.content)

        try:
            output_dict: dict[str, any] = json.loads(response_cleaned)
            break  # Exit loop if JSON is successfully loaded
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON on attempt %d: %s", attempt + 1,
                         response_cleaned)
    else:
        raise ValueError("Max retries reached. JSON could not be parsed.")

    with open(output_file_path, 'w') as fp:
        json.dump(output_dict, fp, indent=4)

    return output_file_path


def parse_tasks_file(task_file: gr.File) -> list[tuple[int, str]]:
    """Parse task file and return a list of tasks."""
    if task_file.name.endswith('.csv'):
        df = pd.read_csv(task_file.name)
    elif task_file.name.endswith('.xlsx'):
        df = pd.read_excel(task_file.name)
    else:
        logger.error("Unsupported file type: %s", task_file.name)
        raise ValueError(
            "Unsupported file type. Please upload a .csv or .xlsx file.")

    return [(
        index,
        f"Task description: {row['Task Description']}, Amount: {row['Amount']}"
    ) for index, row in df.iterrows()]


def llm_check_tasks(json_data: str, tasks: list[tuple[int, str]],
                    progress: gr.Progress) -> list[dict[str, any]]:
    """Batch execute tasks, retrying failed JSON loads."""
    remaining_tasks: list[tuple[int, str]] = tasks
    tasks_decisions: list[dict[str, any]] = []
    processed_tasks: set[int] = set(
    )  # To keep track of tasks that have been processed

    total_tasks = len(tasks)
    progress(0, total=total_tasks)  # Initialize progress bar

    for attempt in range(config["openai"]["max_retries"]):
        responses = langchain.batch(
            inputs=[{
                "system_prompt": config["prompts"]["check_task_system"],
                "user_prompt": f"Contract: {json_data}\n\n\n {task[1]}"
            } for task in remaining_tasks],
            config={"max_concurrency": config["openai"]["max_concurrency"]})

        new_remaining_tasks: list[tuple[int, str]] = []
        for i, response in enumerate(responses):
            logger.info("Running task: %s", remaining_tasks[i][1])
            cleaned_response: str = clean_json_prefix_suffix(response.content)

            try:
                response_data: dict[str, any] = json.loads(cleaned_response)
                task_decision: dict[str, any] = {
                    'Task Number': remaining_tasks[i][0],
                    'Task': remaining_tasks[i][1]
                }
                task_decision.update(response_data)
                tasks_decisions.append(task_decision)
                processed_tasks.add(
                    remaining_tasks[i][0])  # Mark this task as processed

                progress(len(processed_tasks) / total_tasks)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode JSON response for task %d on attempt %d: %s",
                    remaining_tasks[i][0], attempt + 1, cleaned_response)
                new_remaining_tasks.append(remaining_tasks[i])

        if not new_remaining_tasks:
            break
        remaining_tasks = new_remaining_tasks

    # Handle tasks that failed even after all retries
    for task in remaining_tasks:
        if task[0] not in processed_tasks:  # Only add if the task hasn't been successfully processed
            logger.error(
                "Max retries reached for task %d. No valid JSON response obtained.",
                task[0])
            tasks_decisions.append({
                'Task Number':
                task[0],
                'Task':
                task[1],
                'Error':
                'Failed to parse JSON after maximum retries'
            })

    tasks_decisions.sort(key=lambda x: x['Task Number'])
    return tasks_decisions


def handle_tasks(contract_file_path: str, tasks_file: gr.File,
                 output_file_path: str, progress: gr.Progress) -> str:
    """Handle task file processing and save results to CSV."""
    tasks: list[tuple[int, str]] = parse_tasks_file(tasks_file)
    with open(contract_file_path, 'r') as fp:
        contract_json_data: str = fp.read()

    tasks_decisions: list[dict[str, any]] = llm_check_tasks(
        json_data=contract_json_data, tasks=tasks, progress=progress)

    df = pd.DataFrame(tasks_decisions)
    df.to_csv(output_file_path)
    return output_file_path


def handle_files_upload(
    contract_file: gr.File, tasks_file: gr.File | None,
    progress=gr.Progress()) -> list[str | None]:
    """Handle file uploads and task processing."""
    progress(0)

    if contract_file.name.endswith('.json'):
        output_json_contract_path: str = contract_file.name
    else:
        output_json_contract_path: str = os.path.splitext(
            contract_file.name)[0] + '.json'
        run_llm_to_get_json_from_contract(contract_file,
                                          output_json_contract_path)

    output_csv_tasks_path: str | None = None

    if tasks_file is not None:
        output_csv_tasks_path = handle_tasks(output_json_contract_path,
                                             tasks_file,
                                             config["paths"]["output_file"],
                                             progress)

    logger.info('The output has been generated 🍺🍺🍺')

    # Ensure two outputs are always returned
    return [output_json_contract_path, output_csv_tasks_path]


def main() -> None:
    global config
    config = load_config()
    setup_logging()
    initialize_llm()

    css: str = """
    #clear-btn {
        display: none !important;
    }
    """

    # Add submit button by setting live=False
    interface = gr.Interface(
        fn=lambda contract_file, tasks_file=None: handle_files_upload(
            contract_file, tasks_file),
        inputs=["file", "file"],
        outputs=[
            gr.File(label="Summarized Contract"),
            gr.File(label="Checked Tasks")
        ],
        title=config["interface"]["title"],
        description=config["interface"]["description"],
        allow_flagging='never',
        css=css,
        live=False  # Disable live updates to show submit button
    )

    interface.launch()


if __name__ == "__main__":
    main()
