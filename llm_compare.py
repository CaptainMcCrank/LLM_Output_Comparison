#!/usr/bin/env python3
import os
import sys
import argparse
import time
import hashlib
from datetime import datetime
from pathlib import Path
import openai
from anthropic import Anthropic
import google.generativeai as genai
#from groq import Groq

# ———————————————
# 1. Configure clients from ENV
# ———————————————
# Remove the old openai configuration
# openai.api_key = os.getenv("OPENAI_API_KEY")  # This is deprecated
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ———————————————
# 2. Utility functions
# ———————————————
def get_md5_hash(content: str) -> str:
    """Get MD5 hash of content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_file_info(filepath: str, prompt: str = None) -> dict:
    """Get file information including path, hash, and modification time."""
    if not filepath:
        # For command line or user input, include the actual prompt
        return {
            'source_path': 'Command line input',
            'source_prompt': prompt or 'N/A',
            'md5_hash': get_md5_hash(prompt) if prompt else 'N/A',
            'last_modified': 'N/A'
        }
    
    path = Path(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {
        'source_path': str(path.absolute()),
        'source_prompt': None,  # Will be read from file
        'md5_hash': get_md5_hash(content),
        'last_modified': datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }

def print_status(message: str, status: str = "INFO"):
    """Print status message with visual flair."""
    colors = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green
        "ERROR": "\033[91m",     # Red
        "WARNING": "\033[93m",   # Yellow
        "WAITING": "\033[96m"    # Cyan
    }
    reset = "\033[0m"
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"{colors.get(status, '')}{timestamp} [{status}] {message}{reset}")

class AnimationController:
    def __init__(self):
        self.stop_animation = False
        self.thread = None
    
    def start(self, message: str):
        """Start animated waiting message."""
        frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        import threading
        import time
        
        def animate():
            i = 0
            while not self.stop_animation:
                print(f"\r\033[96m{frames[i % len(frames)]} {message}\033[0m", end='', flush=True)
                time.sleep(0.1)
                i += 1
        
        self.stop_animation = False
        self.thread = threading.Thread(target=animate, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop animation and clear line."""
        self.stop_animation = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)  # Don't wait forever
        print('\r' + ' ' * 60 + '\r', end='', flush=True)

def read_markdown_file(filepath: str) -> str:
    """Read content from a markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print_status(f"File '{filepath}' not found.", "ERROR")
        sys.exit(1)
    except Exception as e:
        print_status(f"Error reading file '{filepath}': {e}", "ERROR")
        sys.exit(1)

def get_prompt_and_source() -> tuple:
    """Get the prompt from command line args, file, or user input."""
    parser = argparse.ArgumentParser(description='Compare LLM outputs across multiple providers')
    parser.add_argument('prompt', nargs='*', help='Prompt text or markdown file path')
    parser.add_argument('--console', '-c', action='store_true', help='Output to console instead of files')
    parser.add_argument('--output-dir', '-o', default='.', help='Directory to store output files (default: current directory)')
    args = parser.parse_args()
    
    if not args.prompt:
        # No arguments - prompt user for input
        return input("Enter your prompt: "), None, args.output_dir
    
    first_arg = args.prompt[0]
    
    if first_arg.endswith('.md') or first_arg.endswith('.markdown'):
        # First argument is a markdown file
        return read_markdown_file(first_arg), first_arg, args.output_dir
    else:
        # Treat all arguments as the prompt text
        return " ".join(args.prompt), None, args.output_dir

def create_output_directory(base_dir: str, timestamp: str) -> Path:
    """Create timestamped output directory."""
    output_dir = Path(base_dir) / f"llm_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print_status(f"Created output directory: {output_dir}", "INFO")
    return output_dir

def copy_source_file(source_file: str, output_dir: Path) -> str:
    """Copy source markdown file to output directory."""
    if not source_file:
        return None
    
    source_path = Path(source_file)
    dest_path = output_dir / f"original_prompt_{source_path.name}"
    
    try:
        import shutil
        shutil.copy2(source_file, dest_path)
        print_status(f"Copied source file to: {dest_path.name}", "INFO")
        return str(dest_path)
    except Exception as e:
        print_status(f"Failed to copy source file: {e}", "WARNING")
        return None

def save_response_to_file(provider: str, response: str, metadata: dict, output_dir: Path, file_info: dict, original_prompt: str = None):
    """Save response to a markdown file with metadata in the output directory."""
    filename = f"{provider}_Response.md"
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {provider} Response\n\n")
        f.write(f"## Query Metadata\n\n")
        f.write(f"- **Model**: {metadata['model']}\n")
        f.write(f"- **Query Timestamp**: {metadata['query_timestamp']}\n")
        f.write(f"- **Response Time**: {metadata['response_time']:.2f} seconds\n")
        f.write(f"- **Temperature**: {metadata.get('temperature', 'N/A')}\n")
        f.write(f"- **Input Tokens**: {metadata.get('input_tokens', 'N/A')}\n")
        f.write(f"- **Output Tokens**: {metadata.get('output_tokens', 'N/A')}\n\n")
        f.write(f"## Source Information\n\n")
        f.write(f"- **Source Path**: {file_info['source_path']}\n")
        
        # If it's command line input, show the actual prompt
        if file_info['source_prompt']:
            f.write(f"- **Original Prompt**: {file_info['source_prompt']}\n")
        
        f.write(f"- **MD5 Hash**: {file_info['md5_hash']}\n")
        f.write(f"- **Last Modified**: {file_info['last_modified']}\n\n")
        f.write(f"## Response\n\n")
        f.write(response)
    
    print_status(f"Response saved to {filename}", "SUCCESS")
    return str(filepath)

# ———————————————
# 3. Define per-model query functions
# ———————————————
def query_chatgpt(prompt: str) -> dict:
    model = "gpt-4o-mini"
    temperature = 0.7
    timeout = 120  # 2 minutes timeout
    
    print_status(f"Querying ChatGPT ({model})...", "WAITING")
    animation = AnimationController()
    animation.start("Waiting for ChatGPT response...")
    
    start_time = time.time()
    query_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Create OpenAI client with timeout
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=timeout
        )
        
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        
        response_time = time.time() - start_time
        animation.stop()
        
        print_status(f"ChatGPT response received ({response_time:.2f}s)", "SUCCESS")
        
        return {
            'response': resp.choices[0].message.content.strip(),
            'metadata': {
                'model': model,
                'query_timestamp': query_timestamp,
                'response_time': response_time,
                'temperature': temperature,
                'input_tokens': resp.usage.prompt_tokens if resp.usage else 'N/A',
                'output_tokens': resp.usage.completion_tokens if resp.usage else 'N/A'
            }
        }
    except Exception as e:
        animation.stop()
        raise e

def query_claude(prompt: str) -> dict:
    model = "claude-3-5-sonnet-20241022"
    temperature = 0.7
    max_tokens = 1000
    
    print_status(f"Querying Claude ({model})...", "WAITING")
    animation = AnimationController()
    animation.start("Waiting for Claude response...")
    
    start_time = time.time()
    query_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        resp = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_time = time.time() - start_time
        animation.stop()
        
        print_status(f"Claude response received ({response_time:.2f}s)", "SUCCESS")
        
        return {
            'response': resp.content[0].text.strip(),
            'metadata': {
                'model': model,
                'query_timestamp': query_timestamp,
                'response_time': response_time,
                'temperature': temperature,
                'input_tokens': resp.usage.input_tokens if resp.usage else 'N/A',
                'output_tokens': resp.usage.output_tokens if resp.usage else 'N/A'
            }
        }
    except Exception as e:
        animation.stop()
        raise e

def query_gemini(prompt: str) -> dict:
    model_name = "gemini-2.0-flash-exp"
    temperature = 0.7
    
    print_status(f"Querying Gemini ({model_name})...", "WAITING")
    animation = AnimationController()
    animation.start("Waiting for Gemini response...")
    
    start_time = time.time()
    query_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        
        response_time = time.time() - start_time
        animation.stop()
        
        print_status(f"Gemini response received ({response_time:.2f}s)", "SUCCESS")
        
        return {
            'response': resp.text.strip(),
            'metadata': {
                'model': model_name,
                'query_timestamp': query_timestamp,
                'response_time': response_time,
                'temperature': temperature,
                'input_tokens': resp.usage_metadata.prompt_token_count if resp.usage_metadata else 'N/A',
                'output_tokens': resp.usage_metadata.candidates_token_count if resp.usage_metadata else 'N/A'
            }
        }
    except Exception as e:
        animation.stop()
        raise e

def query_grok(prompt: str) -> dict:
    model = "llama-3.3-70b-versatile"
    temperature = 0.7
    
    print_status(f"Querying Grok ({model})...", "WAITING")
    animation = AnimationController()
    animation.start("Waiting for Grok response...")
    
    start_time = time.time()
    query_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature
        )
        
        response_time = time.time() - start_time
        animation.stop()
        
        print_status(f"Grok response received ({response_time:.2f}s)", "SUCCESS")
        
        return {
            'response': chat.choices[0].message.content.strip(),
            'metadata': {
                'model': model,
                'query_timestamp': query_timestamp,
                'response_time': response_time,
                'temperature': temperature,
                'input_tokens': chat.usage.prompt_tokens if chat.usage else 'N/A',
                'output_tokens': chat.usage.completion_tokens if chat.usage else 'N/A'
            }
        }
    except Exception as e:
        animation.stop()
        raise e

def convert_to_html_and_open(markdown_files: list, output_dir: Path):
    """Convert markdown files to HTML using pandoc and open them."""
    import subprocess
    import shutil
    
    # Check if pandoc is installed
    if not shutil.which("pandoc"):
        print_status("Pandoc not found. Install with: sudo apt install pandoc (Ubuntu) or brew install pandoc (Mac)", "WARNING")
        return
    
    html_files = []
    
    for md_file in markdown_files:
        md_path = Path(md_file)
        html_file = output_dir / md_path.name.replace('.md', '.html')
        
        try:
            # Convert markdown to HTML
            print_status(f"Converting {md_path.name} to HTML...", "INFO")
            subprocess.run([
                'pandoc', str(md_path), 
                '-s',  # standalone HTML
                '-o', str(html_file)
            ], check=True, capture_output=True)
            
            html_files.append(str(html_file))
            print_status(f"Created {html_file.name}", "SUCCESS")
            
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to convert {md_path.name}: {e}", "ERROR")
        except Exception as e:
            print_status(f"Error converting {md_path.name}: {e}", "ERROR")
    
    # Open all HTML files
    if html_files:
        print_status(f"Opening {len(html_files)} HTML files...", "INFO")
        
        for html_file in html_files:
            try:
                # Use xdg-open on Linux, open on Mac, start on Windows
                if shutil.which("xdg-open"):
                    subprocess.Popen(['xdg-open', html_file])
                elif shutil.which("open"):  # macOS
                    subprocess.Popen(['open', html_file])
                elif shutil.which("start"):  # Windows
                    subprocess.Popen(['start', html_file], shell=True)
                else:
                    print_status(f"Cannot auto-open {Path(html_file).name}. Open manually in browser.", "WARNING")
                    
            except Exception as e:
                print_status(f"Failed to open {Path(html_file).name}: {e}", "ERROR")
                
        print_status("HTML files opened in default browser", "SUCCESS")
# ———————————————
# 4. Main execution
# ———————————————
def main():
    # Check if console output flag is set
    console_output = '--console' in sys.argv or '-c' in sys.argv
    
    prompt, source_file, base_output_dir = get_prompt_and_source()
    file_info = get_file_info(source_file, prompt)
    
    # Generate timestamp for directory naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print_status("Starting LLM comparison...", "INFO")
    print(f"\nPrompt:\n{prompt}\n{'='*60}\n")
    
    # Create output directory if not using console output
    if not console_output:
        output_dir = create_output_directory(base_output_dir, timestamp)
        
        # Copy source file if it exists
        if source_file:
            copy_source_file(source_file, output_dir)
    
    providers = [
        ("ChatGPT", query_chatgpt),
        ("Claude", query_claude),
        ("Gemini", query_gemini),
        # ("Grok", query_grok),
    ]
    
    markdown_files = []
    
    for name, fn in providers:
        try:
            result = fn(prompt)
            
            if console_output:
                print(f"\n--- {name} ---\n{result['response']}\n")
            else:
                filepath = save_response_to_file(name, result['response'], result['metadata'], output_dir, file_info, prompt)
                markdown_files.append(filepath)
                
        except Exception as e:
            print_status(f"{name} failed: {e}", "ERROR")
            if console_output:
                print(f"\n--- {name} ERROR ---\n{e!r}\n")
    
    if not console_output and markdown_files:
        print_status(f"All responses saved to: {output_dir}", "SUCCESS")
        
        # Convert to HTML and open
        print_status("Converting to HTML and opening files...", "INFO")
        convert_to_html_and_open(markdown_files, output_dir)

if __name__ == "__main__":
    main()