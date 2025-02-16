# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pydantic",
#   "uvicorn[standard]",
#   "fastapi",
#   "numpy",
#   "fastapi[all]",
#   "Pillow"
# ]
# ///

import httpx
import os
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import csv
import json

# move it to environment variable
#api_base = os.environ["AIPROXY_URL"] if "AIPROXY_URL" in os.environ else 'https://aiproxy.sanand.workers.dev/openai'
api_base = 'https://aiproxy.sanand.workers.dev/openai'
api_key = os.environ["AIPROXY_TOKEN"] if "AIPROXY_TOKEN" in os.environ else 'sk-0f2b3b4c-4b3f-4f2d-8f3d-2f3b4c4f3d4f'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run")
async def run_task(task: str):
    print("task: ", task)
    if(task):
        response = query_gpt(task, tools)
        if(response and response["tool_calls"]):
            task_name = response["tool_calls"][0]["function"]["name"]
            task_args = json.loads(response["tool_calls"][0]["function"]["arguments"])
            return execute_task(task_name, task_args)
        
    return {"message": "What task you want to run?"}

@app.get("/read")
async def read_file(path: str):
    if(path):
        try:
            with open(f"./{path}", 'r') as f:
                text = f.read()
        except:
            return {"message": "File not found"}
        return text
    return {"message": "What file you want to read?"}

def query_gpt(user_input: str, tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        print(f"END POINT : {api_base}/v1/chat/completions")
        response = httpx.post(
            f"{api_base}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": user_input}
                ],
                "tools": tools,
                "tool_choice": "auto",
            },
        )
    except httpx.HTTPError as err:
        print(err)
        return None
    print(response.status_code)# show the status code and message
    print(response.text)
    return response.json()["choices"][0]["message"]

def gen_embedding(data):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": data,
        "model": "text-embedding-3-small",
        "encoding_format": "float"
    }
    response = httpx.post(f"{api_base}/v1/embeddings", headers=headers, json=payload)
    print(response)
    response.raise_for_status()
    result = response.json()
    return result

def find_most_similar_comments(comments: list[str]):
    embeddings = gen_embedding(comments)["data"][0]["embedding"]
    vectors = np.array(embeddings, dtype=np.float32)

    # Normalize vectors (cosine similarity formula: cos(θ) = A·B / (||A|| * ||B||))
    norms = np.linalg.norm(vectors, axis=0, keepdims=True)
    normalized_vectors = vectors / norms

    # Compute cosine similarity between all pairs of comments
    similarities = np.dot(normalized_vectors, normalized_vectors.T)

    # Find the indices of the top N most similar comments (excluding self-similarity)
    most_similar_idxs = np.argsort(similarities[~np.eye(similarities.shape[0], dtype=bool)].reshape(similarities.shape[0], -1), axis=1)[:, -n:]

    # Return the top N most similar comments and their similarity scores
    most_similar_comments = [comments[idx] for idx in most_similar_idxs[0]]
    most_similar_scores = [similarities[0, idx] for idx in most_similar_idxs[0]]

    return most_similar_comments, most_similar_scores

tools = [
    {
        "type": "function",
        "function": {
            "name": "a1_run_remote_script",
            "description": "Run <url> with <email> as the only arguement",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Remote URL of the script to execute"},
                    "email": {"type": "string", "description": "email address, only parameter of the script"}
                },
                "required": ["url","email"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a2_format_content_with_prettier",
            "description": "Format the content of <file> using prettier@<version>",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File with path that needs to be formatted"},
                    "version": {"type": "string", "description": "Version of prettier to use for formatting"}
                },
                "required": ["file_path", "version"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a3_count_days_in_file",
            "description": "Count the number of <day> in the <input_file> that contains a list of dates per line, Write the number to <output_file>",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "file that contains list of dates per line"},
                    "output_file": {"type": "string", "description": "write the number of count of days in this file"},
                    "day": {"type": "string", "description": "what day of the week to count"}
                },
                "required": ["input_file","output_file", "day"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a4_sort_json_based_on_value",
            "description": "Sort the array of objects in <input_file> by <field_a> and then by <field_b> and save results in <output_file>",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "file that contains array of objects"},
                    "output_file": {"type": "string", "description": "write the sorted array in this file"},
                    "field_a": {"type": "string", "description": "sort by this field value first"},
                    "field_b": {"type": "string", "description": "sort by this field value next"}
                },
                "required": ["input_file","output_file", "field_a", "field_b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a5_specific_logs",
            "description": "Write the first line of the <limit> most recent .log file in <log_dir> to <output_file>, most recent first",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_dir": {"type": "string", "description": "folder contains all the logs files"},
                    "output_file": {"type": "string", "description": "write the result in this file"},
                    "limit": {"type": "string", "description": "number of log files"},
                    "order_by": {"type": "string", "description": "most or least recent in output file"}
                },
                "required": ["log_dir","output_file", "limit", "order_by"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a6_create_index_of_markdown_files",
            "description": "Find all markdown files in <md_folder>, For each extract first occurance of <h_tag>. create index file <index_file> that contains file_name,<h_tag> as key value pair",
            "parameters": {
                "type": "object",
                "properties": {
                    "md_folder": {"type": "string", "description": "folder contains all the logs files"},
                    "output_file": {"type": "string", "description": "write the result in this file"},
                    "limit": {"type": "string", "description": "number of log files"},
                    "order_by": {"type": "string", "description": "most or least recent in output file"}
                },
                "required": ["md_folder","output_file", "limit", "order_by"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a7_find_text_in_email",
            "description": "From email message in file <email_file>, find sender/reciever/subject and write it to <output_file>",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_file": {"type": "string", "description": "file contains email message"},
                    "output_file": {"type": "string", "description": "write the result in this file"},
                    "find": {"type": "string", "description": "key information to find in email ex. sender, reciever, subject"}
                },
                "required": ["email_file","output_file", "find"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a8_extract_text_from_image",
            "description": "<img_file> contains <find>, get the <find> and write it to <output_file>",
            "parameters": {
                "type": "object",
                "properties": {
                    "img_file": {"type": "string", "description": "image file contains text and number"},
                    "output_file": {"type": "string", "description": "write the searched text or number to this file"},
                    "find": {"type": "string", "description": "search text or number mentioned here"}
                },
                "required": ["img_file","output_file", "find"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a9_find_most_similar_comments",
            "description": "<input_file> contains list of comments/texts one per line, find most similar pair of comments/texts and write it to <output_file>",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "It contains list of comments/texts per line"},
                    "output_file": {"type": "string", "description": "write the most similar pair of comments/text in this file"}
                },
                "required": ["input_file","output_file"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "a10_total_sales_of_ticket",
            "description": "<db_file> has a <table> with columns type, units, and price, find total sales for a <type>",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_file": {"type": "string", "description": "sqlite database file"},
                    "output_file": {"type": "string", "description": "sql query result to writ here"},
                    "ticket_type": {"type": "string", "description": "type of ticket"}
                },
                "required": ["db_file","output_file", "ticket_type"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b1_no_access_outside_data",
            "description": "Folder and file path not starting with /data",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File of outside /data folder"}
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b2_no_delete",
            "description": "Delete not allowed",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Files to delete"}
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b3_fetch_data_from_api",
            "description": "Fetch data from an API",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_end_point": {"type": "string", "description": "API endpoint to fetch data"}
                },
                "required": ["api_end_point"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b4_clone_repo_commit",
            "description": "Clone a git repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "git repository URL"}
                },
                "required": ["repo_url"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b5_run_sql_query",
            "description": "Run SQL query on sqlite db file",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_file": {"type": "string", "description": "sqlite db file"},
                    "query": {"type": "string", "description": "sql statement"}
                },
                "required": ["db_file","query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b6_extract_data_site",
            "description": "Extract data from a site",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the site"}
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b7_compress_or_resize",
            "description": "Compress and resize Image file",
            "parameters": {
                "type": "object",
                "properties": {
                    "img_file": {"type": "string", "description": "Image file to process"},
                    "max_size": {"type": "number", "description": "Max size of the image"}
                },
                "required": ["img_file","max_size"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b8_transcribe_mp3_audio",
            "description": "Generate transcript of MP3 audio file",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_file": {"type": "string", "description": "mp3 audio file to transcribe"}
                },
                "required": ["audio_file"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b9_markdown_to_html",
            "description": "Convert markdown file to html",
            "parameters": {
                "type": "object",
                "properties": {
                    "md_file": {"type": "string", "description": "markdown file"}
                },
                "required": ["md_file"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "b10_write_API_endpoint",
            "description": "Generate an API endpoint",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "Instruction for API end point"}
                },
                "required": ["instruction"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]



# Phase A Tasks
def a1_run_remote_script(url: str, email: str):
    print("url: ", url)
    print("email: ", email)
    return {"name": "task_A1_generate_data", "arguements": {"url": url, "email": email}}

def a2_format_content_with_prettier(file_path: str, version: str):
    return {"name": "fomat_content", "arguements": {"file path": file_path, "version": version}}

def a3_count_days_in_file(input_file: str, output_file: str, day: str):
    return {"name": "count_days", "arguements": {"input_file": input_file, "output_file": output_file, "day": day}}

def a4_sort_json_based_on_value(input_file: str, output_file: str, field_a: str, field_b: str):
    return {"name": "sort_by", "arguements": {"input_file": input_file, "output_file": output_file, "sort_by_a": sort_by_a, "sort_by_b": sort_by_b}}

def a5_specific_logs(log_dir: str, output_file: str, limit: int, order_by: str):
    return {"name": "recent_logs", "arguements": {"log_dir": log_dir, "output_file": output_file, "limit": limit, "order_by": order_by}}

def a6_create_index_of_markdown_files(md_folder: str,index_file: str, h_tag: str):
    index_file=f"{index_file}"
    return {"name": "create_index_for_markdown_files", "arguements": {"folder_md": folder_md, "index_file": index_file, "h_tag": h_tag}}

def a7_find_text_in_email(email_file: str, output_file: str, find: str):
    return {"name": "find_text_in_email", "arguements": {"email_file": email_file, "output_file": output_file,"find": find}}

def a8_extract_text_from_image(img_file: str, output_file: str, find: str):
    return {"name": "extract_text_from_image", "arguements": {"img_file": img_file, "output_file": output_file, "find": find}}

def a9_find_most_similar_comments(input_file: str, output_file: str):
    return {"name": "find_most_similar_comments", "arguements": {"input_file": input_file, "output_file": output_file}}

def a10_total_sales_of_ticket(db_file: str, output_file: str, ticket_type: str): 
    return {"name": "total_sales", "arguements": {"db_file": db_file, "output_file": output_file,"ticket_type": ticket_type}}

# Phase B Tasks

def b1_no_access_outside_data(file_path: str):
    return {"name": "no_access_outside_data", "arguements": {"file_path": file_path}}

def b2_no_delete(file_path: str):
    return {"name": "no_delete", "arguements": {"file_path": file_path}}

def b3_fetch_data_from_api(api_end_point: str):
    return {"name": "fetch_data_from_api", "arguements": {"api_end_point": api_end_point}}

def b4_clone_repo_commit(repo_url: str):
    return {"name": "clone_repo_commit", "arguements": {"repo_url": repo_url}}

def b5_run_sql_query(db_file: str, query: str):
    return {"name": "run_sql_query", "arguements": {"db_file": db_file, "query": query}}

def b6_extract_data_site(url: str):
    return {"name": "extract_data_site", "arguements": {"url": url}}

def b7_compress_or_resize(img_file: str, max_size: int):
    return {"name": "compress_or_resize", "arguements": {"img_file": img_file, "max_size": max_size}}

def b8_transcribe_mp3_audio(audio_file: str):
    return {"name": "transcribe_mp3_audio", "arguements": {"audio_file": audio_file}}

def b9_markdown_to_html(md_file: str):
    return {"name": "markdown_to_html", "arguements": {"md_file": md_file}}

def b10_write_API_endpoint(instruction: str):
    return {"name": "write_API_endpoint", "arguements": {"instruction": instruction}}


# create a map of functions starting wit a and b
task_map = {
    func_name: globals()[func_name]
    for func_name in globals()
    if func_name.startswith("a") or func_name.startswith("b")
}

def execute_task(func_name, args):
    task = task_map.get(func_name)
    if task:
        return task(**args)
    return "Invalid task"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
