#!/usr/bin/env python3
import os
import sys
import argparse
import time
import hashlib
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any
import openai
from anthropic import Anthropic
import google.generativeai as genai
#from groq import Groq

# ———————————————
# Configuration System
# ———————————————
@dataclass
class ResponseMetadata:
    """Standardized metadata for LLM responses."""
    model: str
    provider: str
    query_timestamp: str
    response_time: float
    temperature: float
    max_tokens: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_estimate: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass 
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    metadata: ResponseMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return asdict(self)

# ———————————————
@dataclass
class ModelConfig:
    """Configuration for a specific LLM provider."""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = 1000
    timeout: int = 120
    enabled: bool = True

@dataclass
class LLMCompareConfig:
    """Main configuration for LLM comparison tool."""
    # Model configurations
    chatgpt: ModelConfig
    claude: ModelConfig
    gemini: ModelConfig
    grok: ModelConfig
    
    # Global settings
    default_temperature: float = 0.7
    default_max_tokens: int = 1000
    default_timeout: int = 120
    
    # Output settings
    auto_open_html: bool = True
    generate_summary: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMCompareConfig':
        """Create config from dictionary."""
        # Convert nested dicts to ModelConfig objects
        for key in ['chatgpt', 'claude', 'gemini', 'grok']:
            if key in data and isinstance(data[key], dict):
                data[key] = ModelConfig(**data[key])
        return cls(**data)

# Configuration presets
PRESET_CONSERVATIVE = LLMCompareConfig(
    chatgpt=ModelConfig("gpt-4o-mini", temperature=0.3, max_tokens=800),
    claude=ModelConfig("claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=800),
    gemini=ModelConfig("gemini-2.0-flash-exp", temperature=0.3, max_tokens=800),
    grok=ModelConfig("llama-3.3-70b-versatile", temperature=0.3, max_tokens=800),
    default_temperature=0.3,
    default_max_tokens=800
)

PRESET_MODERATE = LLMCompareConfig(
    chatgpt=ModelConfig("gpt-4o-mini", temperature=0.7, max_tokens=1000),
    claude=ModelConfig("claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=1000),
    gemini=ModelConfig("gemini-2.0-flash-exp", temperature=0.7, max_tokens=1000),
    grok=ModelConfig("llama-3.3-70b-versatile", temperature=0.7, max_tokens=1000),
    default_temperature=0.7,
    default_max_tokens=1000
)

PRESET_CREATIVE = LLMCompareConfig(
    chatgpt=ModelConfig("gpt-4o-mini", temperature=1.0, max_tokens=1500),
    claude=ModelConfig("claude-3-5-sonnet-20241022", temperature=1.0, max_tokens=1500),
    gemini=ModelConfig("gemini-2.0-flash-exp", temperature=1.0, max_tokens=1500),
    grok=ModelConfig("llama-3.3-70b-versatile", temperature=1.0, max_tokens=1500),
    default_temperature=1.0,
    default_max_tokens=1500
)

PRESETS = {
    'conservative': PRESET_CONSERVATIVE,
    'moderate': PRESET_MODERATE,
    'creative': PRESET_CREATIVE
}

def load_config(config_path: Optional[str] = None, preset: Optional[str] = None) -> LLMCompareConfig:
    """Load configuration from file or preset."""
    if preset and preset in PRESETS:
        print_status(f"Using {preset} preset configuration", "INFO")
        return PRESETS[preset]
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            print_status(f"Loaded configuration from {config_path}", "INFO")
            return LLMCompareConfig.from_dict(config_data)
        except Exception as e:
            print_status(f"Failed to load config from {config_path}: {e}", "ERROR")
            print_status("Using moderate preset instead", "WARNING")
    
    return PRESET_MODERATE

def save_config(config: LLMCompareConfig, config_path: str):
    """Save configuration to file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print_status(f"Configuration saved to {config_path}", "SUCCESS")
    except Exception as e:
        print_status(f"Failed to save config: {e}", "ERROR")

def validate_api_keys():
    """Validate that required API keys are present."""
    required_keys = {
        'OPENAI_API_KEY': 'ChatGPT',
        'ANTHROPIC_API_KEY': 'Claude',
        'GOOGLE_API_KEY': 'Gemini'
    }
    
    missing_keys = []
    for env_key, provider in required_keys.items():
        if not os.getenv(env_key):
            missing_keys.append(f"{env_key} (for {provider})")
    
    if missing_keys:
        print_status("Missing API keys:", "ERROR")
        for key in missing_keys:
            print_status(f"  - {key}", "ERROR")
        print_status("Some providers will be disabled", "WARNING")
        return False
    
    print_status("All API keys validated", "SUCCESS")
    return True

def validate_parameters(config: LLMCompareConfig) -> bool:
    """Validate configuration parameters."""
    issues = []
    
    # Check temperature range (0.0 to 2.0 for most providers)
    for provider_name in ['chatgpt', 'claude', 'gemini', 'grok']:
        provider_config = getattr(config, provider_name)
        if provider_config.temperature < 0.0 or provider_config.temperature > 2.0:
            issues.append(f"{provider_name} temperature ({provider_config.temperature}) must be between 0.0 and 2.0")
        
        if provider_config.max_tokens is not None and provider_config.max_tokens < 1:
            issues.append(f"{provider_name} max_tokens ({provider_config.max_tokens}) must be positive")
            
        if provider_config.timeout < 1:
            issues.append(f"{provider_name} timeout ({provider_config.timeout}) must be at least 1 second")
    
    if issues:
        print_status("Configuration validation failed:", "ERROR")
        for issue in issues:
            print_status(f"  - {issue}", "ERROR")
        return False
    
    print_status("Configuration parameters validated", "SUCCESS")
    return True

def generate_temperature_sweep(min_temp: float, max_temp: float, steps: int) -> list:
    """Generate temperature values for parameter sweep."""
    if steps == 1:
        return [(min_temp + max_temp) / 2]
    
    step_size = (max_temp - min_temp) / (steps - 1)
    return [min_temp + i * step_size for i in range(steps)]

def create_sweep_configs(base_config: LLMCompareConfig, temperatures: list) -> list:
    """Create multiple configurations for temperature sweep."""
    configs = []
    for temp in temperatures:
        # Create a copy of the base config
        import copy
        sweep_config = copy.deepcopy(base_config)
        
        # Update temperature for all providers
        sweep_config.chatgpt.temperature = temp
        sweep_config.claude.temperature = temp
        sweep_config.gemini.temperature = temp
        sweep_config.grok.temperature = temp
        
        configs.append((f"temp_{temp:.1f}", sweep_config))
    
    return configs

def read_batch_prompts(batch_file: str) -> list:
    """Read prompts from batch file."""
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print_status(f"Loaded {len(prompts)} prompts from {batch_file}", "INFO")
        return prompts
    except Exception as e:
        print_status(f"Failed to read batch file {batch_file}: {e}", "ERROR")
        sys.exit(1)

def run_single_comparison(prompt: str, config: LLMCompareConfig, config_name: str, 
                         base_output_dir: str, timestamp: str, console_output: bool) -> dict:
    """Run a single LLM comparison and return results."""
    print_status(f"Running comparison with {config_name} configuration", "INFO")
    
    # Create output directory for this configuration
    if not console_output:
        if config.generate_summary:
            summary = generate_prompt_summary(prompt)
            dir_name = f"llm_comparison_{timestamp}_{config_name}_{summary}"
        else:
            dir_name = f"llm_comparison_{timestamp}_{config_name}"
        
        output_dir = Path(base_output_dir) / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print_status(f"Created output directory: {output_dir}", "INFO")
    
    providers = [
        ("ChatGPT", query_chatgpt, config.chatgpt),
        ("Claude", query_claude, config.claude),
        ("Gemini", query_gemini, config.gemini),
        # ("Grok", query_grok, config.grok),
    ]
    
    results = {}
    markdown_files = []
    
    for name, fn, provider_config in providers:
        if not provider_config.enabled:
            print_status(f"{name} is disabled in configuration", "WARNING")
            continue
            
        result = fn(prompt, provider_config)
        results[name] = result
        
        if console_output:
            if result.metadata.success:
                print(f"\n--- {name} ({config_name}) ---\n{result.content}\n")
            else:
                print(f"\n--- {name} ({config_name}) ERROR ---\n{result.metadata.error_message}\n")
        else:
            file_info = get_file_info(None, prompt)
            filepath = save_response_to_file(result, output_dir, file_info, prompt)
            markdown_files.append(filepath)
    
    return {
        'config_name': config_name,
        'results': results,
        'output_dir': output_dir if not console_output else None,
        'markdown_files': markdown_files
    }

def generate_experiment_summary(all_results: list, base_output_dir: str, timestamp: str) -> str:
    """Generate a summary report for multiple experimental runs."""
    summary_file = Path(base_output_dir) / f"experiment_summary_{timestamp}.md"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# LLM Comparison Experiment Summary\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Configurations Tested**: {len(all_results)}\n\n")
        
        # Performance comparison table
        f.write("## Performance Comparison\n\n")
        f.write("| Configuration | ChatGPT Time | Claude Time | Gemini Time | Avg Time |\n")
        f.write("|---------------|--------------|-------------|-------------|----------|\n")
        
        for exp_result in all_results:
            config_name = exp_result['config_name']
            results = exp_result['results']
            
            times = []
            chatgpt_time = "N/A"
            claude_time = "N/A" 
            gemini_time = "N/A"
            
            if 'ChatGPT' in results and results['ChatGPT'].metadata.success:
                chatgpt_time = f"{results['ChatGPT'].metadata.response_time:.2f}s"
                times.append(results['ChatGPT'].metadata.response_time)
                
            if 'Claude' in results and results['Claude'].metadata.success:
                claude_time = f"{results['Claude'].metadata.response_time:.2f}s"
                times.append(results['Claude'].metadata.response_time)
                
            if 'Gemini' in results and results['Gemini'].metadata.success:
                gemini_time = f"{results['Gemini'].metadata.response_time:.2f}s"
                times.append(results['Gemini'].metadata.response_time)
            
            avg_time = f"{sum(times)/len(times):.2f}s" if times else "N/A"
            
            f.write(f"| {config_name} | {chatgpt_time} | {claude_time} | {gemini_time} | {avg_time} |\n")
        
        # Token usage comparison
        f.write("\n## Token Usage Comparison\n\n")
        f.write("| Configuration | Provider | Input Tokens | Output Tokens | Total Tokens |\n")
        f.write("|---------------|----------|--------------|---------------|-------------|\n")
        
        for exp_result in all_results:
            config_name = exp_result['config_name']
            results = exp_result['results']
            
            for provider, result in results.items():
                if result.metadata.success:
                    input_tokens = result.metadata.input_tokens or 0
                    output_tokens = result.metadata.output_tokens or 0
                    total_tokens = (input_tokens + output_tokens) if input_tokens and output_tokens else "N/A"
                    
                    f.write(f"| {config_name} | {provider} | {input_tokens} | {output_tokens} | {total_tokens} |\n")
        
        # Success rate comparison
        f.write("\n## Success Rate by Provider\n\n")
        success_counts = {}
        total_counts = {}
        
        for exp_result in all_results:
            for provider, result in exp_result['results'].items():
                if provider not in success_counts:
                    success_counts[provider] = 0
                    total_counts[provider] = 0
                
                total_counts[provider] += 1
                if result.metadata.success:
                    success_counts[provider] += 1
        
        for provider in success_counts:
            success_rate = (success_counts[provider] / total_counts[provider]) * 100
            f.write(f"- **{provider}**: {success_counts[provider]}/{total_counts[provider]} ({success_rate:.1f}%)\n")
        
        # Configuration details
        f.write("\n## Configuration Details\n\n")
        for exp_result in all_results:
            config_name = exp_result['config_name']
            f.write(f"### {config_name}\n")
            f.write(f"- **Output Directory**: `{exp_result['output_dir']}`\n")
            f.write(f"- **Files Generated**: {len(exp_result['markdown_files'])}\n\n")
    
    print_status(f"Experiment summary saved to: {summary_file}", "SUCCESS")
    return str(summary_file)

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
    
    # Configuration options
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--preset', choices=['conservative', 'moderate', 'creative'], 
                       help='Use a configuration preset (conservative=0.3, moderate=0.7, creative=1.0)')
    parser.add_argument('--save-config', help='Save current configuration to specified file path')
    
    # Parameter overrides
    parser.add_argument('--temperature', type=float, help='Override temperature for all models')
    parser.add_argument('--max-tokens', type=int, help='Override max tokens for all models')
    parser.add_argument('--timeout', type=int, help='Override timeout for all models (seconds)')
    
    # Experimentation features
    parser.add_argument('--sweep-temperature', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Temperature sweep range (e.g., --sweep-temperature 0.3 1.0)')
    parser.add_argument('--sweep-steps', type=int, default=3, 
                       help='Number of steps in parameter sweep (default: 3)')
    parser.add_argument('--batch-file', help='File containing multiple prompts (one per line)')
    parser.add_argument('--compare-configs', nargs='+', 
                       help='Compare multiple config files (space-separated paths)')
    
    args = parser.parse_args()
    
    if not args.prompt:
        # No arguments - prompt user for input
        return input("Enter your prompt: "), None, args.output_dir, args
    
    first_arg = args.prompt[0]
    
    if first_arg.endswith('.md') or first_arg.endswith('.markdown'):
        # First argument is a markdown file
        return read_markdown_file(first_arg), first_arg, args.output_dir, args
    else:
        # Treat all arguments as the prompt text
        return " ".join(args.prompt), None, args.output_dir, args

def generate_prompt_summary(prompt: str) -> str:
    """Generate a concise summary of the prompt using Claude for directory naming."""
    try:
        summary_prompt = f"""Create a very brief, descriptive summary of this prompt in exactly 10 words or fewer. Use only alphanumeric characters, hyphens, and underscores. Make it suitable for a directory name.

Prompt to summarize:
{prompt[:500]}{"..." if len(prompt) > 500 else ""}

Return only the summary, nothing else."""

        resp = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",  # Use faster, cheaper model for summaries
            max_tokens=50,
            temperature=0.3,  # Lower temperature for consistent naming
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        summary = resp.content[0].text.strip()
        # Clean up the summary for directory naming
        summary = "".join(c if c.isalnum() or c in "-_" else "_" for c in summary)
        summary = "_".join(summary.split())  # Replace multiple underscores with single
        return summary[:50]  # Limit length for filesystem compatibility
        
    except Exception as e:
        print_status(f"Failed to generate prompt summary: {e}", "WARNING")
        return "prompt_summary"

def create_output_directory(base_dir: str, timestamp: str, prompt: str = None) -> Path:
    """Create timestamped output directory with optional prompt summary."""
    if prompt:
        print_status("Generating prompt summary for directory name...", "INFO")
        summary = generate_prompt_summary(prompt)
        dir_name = f"llm_comparison_{timestamp}_{summary}"
    else:
        dir_name = f"llm_comparison_{timestamp}"
    
    output_dir = Path(base_dir) / dir_name
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

def save_response_to_file(llm_response: LLMResponse, output_dir: Path, file_info: dict, original_prompt: str = None):
    """Save LLMResponse to a markdown file with metadata in the output directory."""
    provider = llm_response.metadata.provider
    filename = f"{provider}_Response.md"
    filepath = output_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {provider} Response\n\n")
        
        # Add success/error status
        if not llm_response.metadata.success:
            f.write(f"❌ **ERROR**: {llm_response.metadata.error_message}\n\n")
        
        f.write(f"## Query Metadata\n\n")
        f.write(f"- **Provider**: {llm_response.metadata.provider}\n")
        f.write(f"- **Model**: {llm_response.metadata.model}\n")
        f.write(f"- **Query Timestamp**: {llm_response.metadata.query_timestamp}\n")
        f.write(f"- **Response Time**: {llm_response.metadata.response_time:.2f} seconds\n")
        f.write(f"- **Temperature**: {llm_response.metadata.temperature}\n")
        f.write(f"- **Max Tokens**: {llm_response.metadata.max_tokens or 'N/A'}\n")
        f.write(f"- **Input Tokens**: {llm_response.metadata.input_tokens or 'N/A'}\n")
        f.write(f"- **Output Tokens**: {llm_response.metadata.output_tokens or 'N/A'}\n")
        if llm_response.metadata.cost_estimate:
            f.write(f"- **Estimated Cost**: ${llm_response.metadata.cost_estimate:.4f}\n")
        f.write(f"- **Success**: {'✅' if llm_response.metadata.success else '❌'}\n\n")
        
        f.write(f"## Source Information\n\n")
        f.write(f"- **Source Path**: {file_info['source_path']}\n")
        
        # If it's command line input, show the actual prompt
        if file_info['source_prompt']:
            f.write(f"- **Original Prompt**: {file_info['source_prompt']}\n")
        
        f.write(f"- **MD5 Hash**: {file_info['md5_hash']}\n")
        f.write(f"- **Last Modified**: {file_info['last_modified']}\n\n")
        f.write(f"## Response\n\n")
        f.write(llm_response.content)
    
    print_status(f"Response saved to {filename}", "SUCCESS")
    return str(filepath)

# ———————————————
# 3. Define per-model query functions
# ———————————————
def query_chatgpt(prompt: str, config: ModelConfig) -> LLMResponse:
    """Query ChatGPT with configuration."""
    if not config.enabled:
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="ChatGPT",
            query_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            response_time=0.0,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=None,
            output_tokens=None,
            success=False,
            error_message="ChatGPT is disabled in configuration"
        )
        return LLMResponse(content="", metadata=metadata)
    
    print_status(f"Querying ChatGPT ({config.model_name})...", "WAITING")
    animation = AnimationController()
    animation.start("Waiting for ChatGPT response...")
    
    start_time = time.time()
    query_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Create OpenAI client with timeout
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=config.timeout
        )
        
        # Build request parameters
        request_params = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature
        }
        
        # Add max_tokens if specified
        if config.max_tokens:
            request_params["max_tokens"] = config.max_tokens
        
        resp = client.chat.completions.create(**request_params)
        
        response_time = time.time() - start_time
        animation.stop()
        
        print_status(f"ChatGPT response received ({response_time:.2f}s)", "SUCCESS")
        
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="ChatGPT",
            query_timestamp=query_timestamp,
            response_time=response_time,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=resp.usage.prompt_tokens if resp.usage else None,
            output_tokens=resp.usage.completion_tokens if resp.usage else None
        )
        
        return LLMResponse(
            content=resp.choices[0].message.content.strip(),
            metadata=metadata
        )
        
    except Exception as e:
        animation.stop()
        response_time = time.time() - start_time
        
        # Return error response
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="ChatGPT", 
            query_timestamp=query_timestamp,
            response_time=response_time,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=None,
            output_tokens=None,
            success=False,
            error_message=str(e)
        )
        
        return LLMResponse(content="", metadata=metadata)

def query_claude(prompt: str, config: ModelConfig) -> LLMResponse:
    """Query Claude with configuration."""
    if not config.enabled:
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Claude",
            query_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            response_time=0.0,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=None,
            output_tokens=None,
            success=False,
            error_message="Claude is disabled in configuration"
        )
        return LLMResponse(content="", metadata=metadata)
    
    print_status(f"Querying Claude ({config.model_name})...", "WAITING")
    animation = AnimationController()
    animation.start("Waiting for Claude response...")
    
    start_time = time.time()
    query_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        resp = anthropic_client.messages.create(
            model=config.model_name,
            max_tokens=config.max_tokens or 1000,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_time = time.time() - start_time
        animation.stop()
        
        print_status(f"Claude response received ({response_time:.2f}s)", "SUCCESS")
        
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Claude",
            query_timestamp=query_timestamp,
            response_time=response_time,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=resp.usage.input_tokens if resp.usage else None,
            output_tokens=resp.usage.output_tokens if resp.usage else None
        )
        
        return LLMResponse(
            content=resp.content[0].text.strip(),
            metadata=metadata
        )
        
    except Exception as e:
        animation.stop()
        response_time = time.time() - start_time
        
        # Return error response
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Claude",
            query_timestamp=query_timestamp,
            response_time=response_time,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=None,
            output_tokens=None,
            success=False,
            error_message=str(e)
        )
        
        return LLMResponse(content="", metadata=metadata)

def query_gemini(prompt: str, config: ModelConfig) -> LLMResponse:
    """Query Gemini with configuration."""
    if not config.enabled:
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Gemini",
            query_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            response_time=0.0,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=None,
            output_tokens=None,
            success=False,
            error_message="Gemini is disabled in configuration"
        )
        return LLMResponse(content="", metadata=metadata)
    
    print_status(f"Querying Gemini ({config.model_name})...", "WAITING")
    animation = AnimationController()
    animation.start("Waiting for Gemini response...")
    
    start_time = time.time()
    query_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        model = genai.GenerativeModel(config.model_name)
        
        # Build generation config
        generation_config = genai.types.GenerationConfig(
            temperature=config.temperature
        )
        
        # Add max_output_tokens if specified
        if config.max_tokens:
            generation_config.max_output_tokens = config.max_tokens
        
        resp = model.generate_content(prompt, generation_config=generation_config)
        
        response_time = time.time() - start_time
        animation.stop()
        
        print_status(f"Gemini response received ({response_time:.2f}s)", "SUCCESS")
        
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Gemini",
            query_timestamp=query_timestamp,
            response_time=response_time,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=resp.usage_metadata.prompt_token_count if resp.usage_metadata else None,
            output_tokens=resp.usage_metadata.candidates_token_count if resp.usage_metadata else None
        )
        
        return LLMResponse(
            content=resp.text.strip(),
            metadata=metadata
        )
        
    except Exception as e:
        animation.stop()
        response_time = time.time() - start_time
        
        # Return error response
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Gemini",
            query_timestamp=query_timestamp,
            response_time=response_time,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=None,
            output_tokens=None,
            success=False,
            error_message=str(e)
        )
        
        return LLMResponse(content="", metadata=metadata)

def query_grok(prompt: str, config: ModelConfig) -> LLMResponse:
    """Query Grok with configuration."""
    if not config.enabled:
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Grok",
            query_timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            response_time=0.0,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=None,
            output_tokens=None,
            success=False,
            error_message="Grok is disabled in configuration"
        )
        return LLMResponse(content="", metadata=metadata)
    
    print_status(f"Querying Grok ({config.model_name})...", "WAITING")
    animation = AnimationController()
    animation.start("Waiting for Grok response...")
    
    start_time = time.time()
    query_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Build request parameters
        request_params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": config.model_name,
            "temperature": config.temperature
        }
        
        # Add max_tokens if specified
        if config.max_tokens:
            request_params["max_tokens"] = config.max_tokens
        
        chat = groq_client.chat.completions.create(**request_params)
        
        response_time = time.time() - start_time
        animation.stop()
        
        print_status(f"Grok response received ({response_time:.2f}s)", "SUCCESS")
        
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Grok",
            query_timestamp=query_timestamp,
            response_time=response_time,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=chat.usage.prompt_tokens if chat.usage else None,
            output_tokens=chat.usage.completion_tokens if chat.usage else None
        )
        
        return LLMResponse(
            content=chat.choices[0].message.content.strip(),
            metadata=metadata
        )
        
    except Exception as e:
        animation.stop()
        response_time = time.time() - start_time
        
        # Return error response
        metadata = ResponseMetadata(
            model=config.model_name,
            provider="Grok",
            query_timestamp=query_timestamp,
            response_time=response_time,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            input_tokens=None,
            output_tokens=None,
            success=False,
            error_message=str(e)
        )
        
        return LLMResponse(content="", metadata=metadata)

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
    # Parse arguments and get configuration
    prompt, source_file, base_output_dir, args = get_prompt_and_source()
    
    # Handle experimentation modes
    if args.sweep_temperature or args.batch_file or args.compare_configs:
        run_experiments(args, base_output_dir)
        return
    
    # Standard single comparison mode
    run_single_standard_comparison(prompt, source_file, base_output_dir, args)

def run_experiments(args, base_output_dir: str):
    """Handle experimental runs (sweeps, batches, config comparisons)."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []
    
    # Load base configuration
    config = load_config(args.config, args.preset)
    
    # Validate API keys
    validate_api_keys()
    
    # Handle temperature sweep
    if args.sweep_temperature:
        print_status(f"Starting temperature sweep from {args.sweep_temperature[0]} to {args.sweep_temperature[1]} in {args.sweep_steps} steps", "INFO")
        
        temperatures = generate_temperature_sweep(args.sweep_temperature[0], args.sweep_temperature[1], args.sweep_steps)
        sweep_configs = create_sweep_configs(config, temperatures)
        
        # Get prompt(s)
        if args.batch_file:
            prompts = read_batch_prompts(args.batch_file)
        else:
            prompt_text = " ".join(args.prompt) if args.prompt else input("Enter your prompt: ")
            prompts = [prompt_text]
        
        # Run sweep for each prompt
        for prompt in prompts:
            print_status(f"Processing prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}", "INFO")
            
            for config_name, sweep_config in sweep_configs:
                if not validate_parameters(sweep_config):
                    print_status(f"Skipping invalid configuration {config_name}", "WARNING")
                    continue
                    
                result = run_single_comparison(prompt, sweep_config, config_name, 
                                             base_output_dir, timestamp, args.console)
                all_results.append(result)
    
    # Handle batch processing
    elif args.batch_file:
        print_status("Starting batch processing", "INFO")
        prompts = read_batch_prompts(args.batch_file)
        
        # Apply parameter overrides
        config = apply_parameter_overrides(config, args)
        if not validate_parameters(config):
            sys.exit(1)
        
        for i, prompt in enumerate(prompts, 1):
            config_name = f"batch_{i:03d}"
            print_status(f"Processing batch item {i}/{len(prompts)}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}", "INFO")
            
            result = run_single_comparison(prompt, config, config_name, 
                                         base_output_dir, timestamp, args.console)
            all_results.append(result)
    
    # Handle config comparison
    elif args.compare_configs:
        print_status(f"Comparing {len(args.compare_configs)} configurations", "INFO")
        
        prompt_text = " ".join(args.prompt) if args.prompt else input("Enter your prompt: ")
        
        for config_path in args.compare_configs:
            config_name = Path(config_path).stem
            comparison_config = load_config(config_path)
            
            if not validate_parameters(comparison_config):
                print_status(f"Skipping invalid configuration {config_name}", "WARNING")
                continue
            
            result = run_single_comparison(prompt_text, comparison_config, config_name, 
                                         base_output_dir, timestamp, args.console)
            all_results.append(result)
    
    # Generate experiment summary
    if all_results and not args.console:
        summary_file = generate_experiment_summary(all_results, base_output_dir, timestamp)
        print_status(f"Experiment completed! Summary available at: {summary_file}", "SUCCESS")

def apply_parameter_overrides(config: LLMCompareConfig, args) -> LLMCompareConfig:
    """Apply CLI parameter overrides to configuration."""
    if args.temperature is not None:
        config.chatgpt.temperature = args.temperature
        config.claude.temperature = args.temperature
        config.gemini.temperature = args.temperature
        config.grok.temperature = args.temperature
        
    if args.max_tokens is not None:
        config.chatgpt.max_tokens = args.max_tokens
        config.claude.max_tokens = args.max_tokens
        config.gemini.max_tokens = args.max_tokens
        config.grok.max_tokens = args.max_tokens
        
    if args.timeout is not None:
        config.chatgpt.timeout = args.timeout
        config.claude.timeout = args.timeout
        config.gemini.timeout = args.timeout
        config.grok.timeout = args.timeout
    
    return config

def run_single_standard_comparison(prompt: str, source_file: str, base_output_dir: str, args):
    """Run a single standard comparison (non-experimental)."""
    file_info = get_file_info(source_file, prompt)
    
    # Load configuration
    config = load_config(args.config, args.preset)
    config = apply_parameter_overrides(config, args)
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, args.save_config)
    
    # Validate API keys and parameters
    validate_api_keys()
    if not validate_parameters(config):
        sys.exit(1)
    
    # Generate timestamp for directory naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print_status("Starting LLM comparison...", "INFO")
    print(f"\nPrompt:\n{prompt}\n{'='*60}\n")
    
    # Create output directory if not using console output
    if not args.console:
        output_dir = create_output_directory(base_output_dir, timestamp, prompt if config.generate_summary else None)
        
        # Copy source file if it exists
        if source_file:
            copy_source_file(source_file, output_dir)
    
    providers = [
        ("ChatGPT", query_chatgpt, config.chatgpt),
        ("Claude", query_claude, config.claude),
        ("Gemini", query_gemini, config.gemini),
        # ("Grok", query_grok, config.grok),
    ]
    
    markdown_files = []
    
    for name, fn, provider_config in providers:
        if not provider_config.enabled:
            print_status(f"{name} is disabled in configuration", "WARNING")
            continue
            
        result = fn(prompt, provider_config)
        
        if args.console:
            if result.metadata.success:
                print(f"\n--- {name} ---\n{result.content}\n")
            else:
                print(f"\n--- {name} ERROR ---\n{result.metadata.error_message}\n")
        else:
            filepath = save_response_to_file(result, output_dir, file_info, prompt)
            markdown_files.append(filepath)
    
    if not args.console and markdown_files:
        print_status(f"All responses saved to: {output_dir}", "SUCCESS")
        
        # Convert to HTML and open if enabled
        if config.auto_open_html:
            print_status("Converting to HTML and opening files...", "INFO")
            convert_to_html_and_open(markdown_files, output_dir)

if __name__ == "__main__":
    main()