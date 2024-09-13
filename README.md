# LLM Document Verifier

`llm_document_verifier` is a tool designed to verify document compliance by leveraging OpenAI's API. The tool parses `.docx` files and uses language model capabilities to summarize and answer questions related to task compliance.

## Demo Video

Watch a demonstration of the product's functionality: [YouTube Video](https://youtu.be/0rq1SCvd8-k)

## Hosted Project

The project is hosted on Replit. You can view and run the project here: [Replit Project Link](https://replit.com/@IgorSlinko/llmdocumentverifier?v=1)

### Running on Replit

To run the project on Replit, you need to set an environment variable `OPENAI_API_KEY` with your OpenAI API key.

### Try it Out

You can test the functionality without running it on Replit by visiting this link: [Live Demo](https://370619bc-62e7-4b43-95d2-52bd92cdb03a-00-2scua1hrwws7a.worf.replit.dev/)

**Example inputs and outputs** can be found in `contents` folder.

## How It Works

1. The `.docx` file is parsed using the built-in `Docx2txtLoader` from LangChain.
2. The parsed content is summarized into a JSON format using OpenAI's API, wrapped in LangChain.
3. This JSON file is used to answer questions regarding task compliance.
4. Additionally, the summarized contract in JSON format is returned, which can be uploaded again in the same form to skip the first step of the algorithm.

## Output

The result of the task verification is saved in a file named `checked_tasks.csv`, which includes the following fields:

- **Task Number**: The task number.
- **Task**: The task text and its cost.
- **why_violates**: Explanation of why the task might violate the contract according to the LLM.
- **why_ok**: Explanation of why the task does not violate the contract according to the LLM.
- **clarification_questions**: Questions to ask the executor to clarify compliance with the contract.
- **decision**: The final decision, which can be one of three types: 'ok', 'violates', or 'unclear'.

A JSON file with a summarized version of the contract is also returned, which can be reused to skip the initial step in future checks.

### Key Features

- **Detailed Reasoning**: Fields like `why_violates`, `why_ok`, and `clarification_questions` are included to provide more time for the LLM (tokens in the self-attention layers) to form a correct final response. These fields also help provide a reasoned explanation for each answer.
- **Concurrency and Speed**: The parallelism and speed of responses depend on the OpenAI user tier. The configuration allows you to set `max_concurrency` (currently set to 1).
- **Retries**: If the JSON responses are not parsed correctly, the number of retries can be set using the `max_retries` parameter (currently set to 3).
- **Prompts**: You can check and modify the prompts used in the configuration file.

## Potential Improvements

- **Multiple Runs with Averaging**: Running the LLM multiple times with different random seeds and averaging the decisions could make predictions more consistent.
- **RAG for Large Documents**: Implementing Retrieval-Augmented Generation (RAG) to handle larger documents effectively.
- **Advanced Prompt Engineering**: Using more sophisticated prompt tricks, as the current algorithm processes everything in a single pass.
- **LangSmith for Debugging**: Using LangSmith for debugging OpenAI API requests.
- **Functional Calls for Calculations**: Currently, the decision making process strongly depends on multiplying costs by various coefficients, and sometimes results are inaccurate. It would be beneficial to use a calculator where relevant.
