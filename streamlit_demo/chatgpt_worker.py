# --------------------------------------------------------
# ChatGPT API Worker as Alternative
# --------------------------------------------------------

import argparse
import json
import time
import uuid
from typing import List, Dict, Any

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from utils import build_logger, server_error_msg

worker_id = str(uuid.uuid4())[:6]
logger = build_logger('chatgpt_worker', f'chatgpt_worker_{worker_id}.log')

class ChatGPTWorker:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    def generate_response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate response using ChatGPT API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get('max_new_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "top_p": kwargs.get('top_p', 0.95),
            "stream": True
        }
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    yield json.dumps({'text': full_response, 'error_code': 0}).encode() + b'\0'
                        except json.JSONDecodeError:
                            continue
            
            return full_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"ChatGPT API error: {e}")
            yield json.dumps({'text': f"API Error: {str(e)}", 'error_code': 1}).encode() + b'\0'
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            yield json.dumps({'text': f"Error: {str(e)}", 'error_code': 1}).encode() + b'\0'

app = FastAPI()

@app.post('/worker_generate_stream')
async def generate_stream(request: Request):
    params = await request.json()
    
    # Extract messages from params
    messages = params.get('prompt', [])
    
    # Convert to ChatGPT format
    chatgpt_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            chatgpt_messages.append({"role": "system", "content": msg['content']})
        elif msg['role'] == 'user':
            content = msg['content']
            # Note: ChatGPT API doesn't support images directly in this simple implementation
            # You would need to use vision models like gpt-4o for image analysis
            chatgpt_messages.append({"role": "user", "content": content})
        elif msg['role'] == 'assistant':
            chatgpt_messages.append({"role": "assistant", "content": msg['content']})
    
    generator = worker.generate_response(chatgpt_messages, **params)
    return StreamingResponse(generator)

@app.post('/worker_get_status')
async def get_status(request: Request):
    return {
        'model_names': [worker.model],
        'speed': 1,
        'queue_length': 0,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=40002)
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='ChatGPT model to use')
    args = parser.parse_args()
    
    worker = ChatGPTWorker(args.api_key, args.model)
    logger.info(f'Starting ChatGPT worker with model: {args.model}')
    
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


