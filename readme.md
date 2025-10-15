# LLM Output Comparison Tool

A Python tool for comparing responses across multiple Large Language Model (LLM) providers. Features advanced experimentation capabilities, comprehensive configuration management, and automated performance analysis. Supports ChatGPT (OpenAI), Claude (Anthropic), and Gemini (Google).

## Features

### Core Functionality
- **Multi-Provider Support**: Query ChatGPT, Claude, and Gemini simultaneously
- **Flexible Input**: Command line prompts, markdown files, or interactive input
- **Rich Output Options**: Timestamped markdown files with intelligent directory naming
- **Automatic HTML Generation**: Convert responses to HTML and open in browser
- **Comprehensive Metadata**: Response times, token usage, model versions, costs, and error tracking

### Advanced Features
- **Configuration Management**: JSON config files with presets (conservative, moderate, creative)
- **Parameter Sweeps**: Automatically test multiple temperature/token configurations
- **Batch Processing**: Process multiple prompts from files with performance tracking
- **Config Comparisons**: Compare responses across different configuration settings
- **Experiment Summaries**: Automated performance reports with token usage and success rates
- **Parameter Validation**: Prevents invalid API calls with comprehensive validation
- **Intelligent Error Handling**: Graceful failures with detailed error metadata
- **Smart Directory Naming**: AI-generated descriptive directory names from prompt content

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Git
- Pandoc (for HTML conversion)

### API Keys Required
You'll need API keys from:
- **OpenAI** (for ChatGPT)
- **Anthropic** (for Claude)
- **Google AI** (for Gemini)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llm-comparison-tool.git
cd llm-comparison-tool
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with:
```
openai>=1.0.0
anthropic>=0.3.0
google-generativeai>=0.3.0
```

### 4. Install Pandoc (for HTML conversion)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install pandoc
```

**macOS:**
```bash
brew install pandoc
```

**Windows:**
Download and install from [pandoc.org](https://pandoc.org/installing.html)

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic API Key (get from https://console.anthropic.com/)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Google AI API Key (get from https://aistudio.google.com/app/apikey)
GOOGLE_API_KEY=your-google-ai-key-here
```

### 6. Load Environment Variables

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) or run before each session:

```bash
# Load environment variables
set -a
source .env
set +a
```

Or install python-dotenv and modify the script to load automatically:
```bash
pip install python-dotenv
```

## Getting API Keys

### OpenAI (ChatGPT)
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new secret key
5. Copy the key (starts with `sk-`)

### Anthropic (Claude)
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Go to API Keys section
4. Generate a new API key
5. Copy the key (starts with `sk-ant-`)

### Google AI (Gemini)
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with Google account
3. Create a new API key
4. Copy the generated key

## Usage

### Prerequisites
Always activate your virtual environment and load environment variables:
```bash
source venv/bin/activate
set -a; source .env; set +a
```

### Basic Usage

**Simple prompt comparison:**
```bash
python llm_compare.py "What is artificial intelligence?"
```

**Interactive mode:**
```bash
python llm_compare.py
# You'll be prompted to enter your question
```

**From markdown file:**
```bash
python llm_compare.py my-prompt.md
```

**Console output (no files):**
```bash
python llm_compare.py "Explain quantum computing" --console
```

### Configuration Management

**Using presets:**
```bash
# Conservative responses (temperature=0.3, max_tokens=800)
python llm_compare.py "Write code" --preset conservative

# Moderate responses (temperature=0.7, max_tokens=1000) - default
python llm_compare.py "Write code" --preset moderate

# Creative responses (temperature=1.0, max_tokens=1500)
python llm_compare.py "Write a story" --preset creative
```

**Parameter overrides:**
```bash
# Override temperature for all models
python llm_compare.py "Generate ideas" --temperature 0.8

# Override max tokens and timeout
python llm_compare.py "Long explanation" --max-tokens 2000 --timeout 180

# Combine presets with overrides
python llm_compare.py "Complex task" --preset conservative --max-tokens 1200
```

**Save/load custom configurations:**
```bash
# Save current config to file
python llm_compare.py "test" --preset creative --save-config my-config.json

# Load config from file
python llm_compare.py "test" --config my-config.json
```

### Advanced Experimentation

**Temperature sweeps:**
```bash
# Test temperatures from 0.3 to 1.0 in 3 steps (0.3, 0.65, 1.0)
python llm_compare.py "Write a haiku" --sweep-temperature 0.3 1.0 --sweep-steps 3

# Fine-grained sweep with 5 steps
python llm_compare.py "Creative writing" --sweep-temperature 0.1 0.9 --sweep-steps 5
```

**Batch processing:**
```bash
# Create a file with multiple prompts (one per line)
echo -e "What is AI?\nExplain machine learning\nDescribe neural networks" > prompts.txt

# Process all prompts
python llm_compare.py --batch-file prompts.txt

# Batch with specific configuration
python llm_compare.py --batch-file prompts.txt --preset creative
```

**Configuration comparisons:**
```bash
# Compare multiple config files
python llm_compare.py "Compare responses" --compare-configs config1.json config2.json config3.json

# Compare presets programmatically (save presets first)
python llm_compare.py "test" --preset conservative --save-config conservative.json
python llm_compare.py "test" --preset creative --save-config creative.json
python llm_compare.py "Creative vs Conservative" --compare-configs conservative.json creative.json
```

**Combined experiments:**
```bash
# Temperature sweep with batch processing
python llm_compare.py --sweep-temperature 0.2 0.8 --sweep-steps 4 --batch-file prompts.txt
```

### Command Line Options

#### Basic Options
- `--console, -c`: Display responses in console instead of saving files
- `--output-dir, -o DIR`: Specify output directory (default: current directory)
- `--help, -h`: Show detailed help message

#### Configuration Options
- `--config CONFIG`: Path to JSON configuration file
- `--preset {conservative,moderate,creative}`: Use built-in configuration preset
- `--save-config FILE`: Save current configuration to JSON file

#### Parameter Overrides
- `--temperature TEMP`: Override temperature for all models (0.0-2.0)
- `--max-tokens TOKENS`: Override max tokens for all models (positive integer)
- `--timeout SECONDS`: Override timeout for all models (minimum 1 second)

#### Experimentation Options
- `--sweep-temperature MIN MAX`: Temperature range for parameter sweep
- `--sweep-steps N`: Number of steps in parameter sweep (default: 3)
- `--batch-file FILE`: File containing multiple prompts (one per line)
- `--compare-configs FILE [FILE ...]`: Compare multiple configuration files

### Output Structure

**Standard single comparison:**
```
llm_comparison_20250801_140352_your-prompt-summary/
├── ChatGPT_Response.md
├── Claude_Response.md
├── Gemini_Response.md
├── ChatGPT_Response.html
├── Claude_Response.html
└── Gemini_Response.html
```

**Experimental runs (sweeps, batches, comparisons):**
```
llm_comparison_20250801_140352_temp_0.3_creative-writing/
llm_comparison_20250801_140352_temp_0.6_creative-writing/
llm_comparison_20250801_140352_temp_0.9_creative-writing/
experiment_summary_20250801_140352.md
```

**Each response file includes:**
- Success/error status with visual indicators (✅/❌)
- Comprehensive metadata (model, provider, timestamps, performance)
- Token usage and cost estimates (when available)
- Source information with MD5 hashes for reproducibility
- Full response content

## Example Workflows

### Basic Comparison
```bash
# Simple single comparison
python llm_compare.py "Compare Python and JavaScript for web development"
```
Results in a directory like `llm_comparison_20250801_140352_python-javascript-comparison/` with responses from all providers.

### Research with Different Configurations
```bash
# Test creative vs conservative approaches
python llm_compare.py "Write a marketing slogan" --preset conservative --save-config conservative.json
python llm_compare.py "Write a marketing slogan" --preset creative --save-config creative.json
python llm_compare.py "Marketing slogan comparison" --compare-configs conservative.json creative.json
```
Generates separate directories for each configuration plus an experiment summary.

### Systematic Parameter Testing
```bash
# Test how temperature affects code generation
python llm_compare.py "Write a Python function to sort a list" --sweep-temperature 0.1 0.9 --sweep-steps 5
```
Creates 5 separate comparisons testing temperatures: 0.1, 0.3, 0.5, 0.7, 0.9

### Batch Processing for Analysis
```bash
# Create test prompts
echo -e "Explain recursion\nWhat is machine learning?\nWrite a sorting algorithm\nDescribe neural networks" > research-questions.txt

# Process all with creative settings
python llm_compare.py --batch-file research-questions.txt --preset creative
```
Generates individual comparisons for each prompt plus a comprehensive experiment summary.

### Quality Assurance Testing
```bash
# Test consistency across multiple runs
python llm_compare.py "Generate 5 creative product names" --sweep-temperature 0.8 0.8 --sweep-steps 3
```
Runs the same prompt 3 times with identical settings to test consistency.

## Troubleshooting

### Common Issues

**"Configuration validation failed":**
- Check parameter ranges: temperature (0.0-2.0), max_tokens (positive), timeout (≥1 second)
- Use `--help` to see valid parameter ranges
- Example fix: `--temperature 0.5` instead of `--temperature -1.0`

**"ModuleNotFoundError":**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

**"API Key not found" or authentication errors:**
- Check `.env` file exists with correct API keys
- Load environment variables: `set -a; source .env; set +a`
- Verify variables are loaded: `echo $OPENAI_API_KEY`
- Check API key permissions and quotas

**"Pandoc not found" warning:**
- Install pandoc: `sudo apt install pandoc` (Ubuntu) or `brew install pandoc` (macOS)
- Script continues without HTML generation if pandoc is missing

**Slow performance or timeouts:**
- Increase timeout: `--timeout 300` (5 minutes)
- Check internet connection and API service status
- Large batch files or parameter sweeps take longer

**Experiment summary not generated:**
- Summaries only created for non-console experimental modes
- Ensure you're not using `--console` flag with experiments
- Check write permissions in output directory

**"Failed to generate prompt summary" warnings:**
- Claude API issues when generating directory names
- Script continues with generic "prompt_summary" directory name
- Not critical for functionality

### Debug Mode

For verbose output, modify the script or add debug prints to troubleshoot API issues.

## Advanced Configuration

### Custom Configuration Files

Create sophisticated configurations beyond the built-in presets:

```json
{
  "chatgpt": {
    "model_name": "gpt-4o",
    "temperature": 0.4,
    "max_tokens": 2000,
    "timeout": 180,
    "enabled": true
  },
  "claude": {
    "model_name": "claude-3-5-sonnet-20241022",
    "temperature": 0.6,
    "max_tokens": 1800,
    "timeout": 200,
    "enabled": true
  },
  "gemini": {
    "model_name": "gemini-1.5-pro",
    "temperature": 0.5,
    "max_tokens": 1500,
    "timeout": 150,
    "enabled": false
  },
  "grok": {
    "model_name": "llama-3.3-70b-versatile",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 120,
    "enabled": false
  },
  "default_temperature": 0.5,
  "default_max_tokens": 1500,
  "default_timeout": 150,
  "auto_open_html": false,
  "generate_summary": true
}
```

### Experiment Design Patterns

**A/B Testing:**
```bash
# Create two configurations for comparison
python llm_compare.py "test" --preset conservative --save-config version-a.json
python llm_compare.py "test" --preset creative --save-config version-b.json

# Test across multiple prompts
python llm_compare.py --batch-file test-prompts.txt --compare-configs version-a.json version-b.json
```

**Performance Benchmarking:**
```bash
# Test different models/providers systematically
# Create configs with different model combinations
python llm_compare.py "benchmark prompt" --sweep-temperature 0.3 0.7 --sweep-steps 3
```

**Quality Analysis:**
```bash
# Run multiple iterations to test consistency
python llm_compare.py "Quality test prompt" --sweep-temperature 0.7 0.7 --sweep-steps 5
```

### Extending the Tool

**Adding New Providers:**
1. Install provider SDK: `pip install provider-package`
2. Add to imports and configure client
3. Create `query_newprovider()` function following existing pattern
4. Add to ModelConfig and LLMCompareConfig classes
5. Update providers list in comparison functions

**Custom Metrics:**
- Add fields to ResponseMetadata class
- Implement cost calculation functions
- Extend experiment_summary generation

**Integration with Analysis Tools:**
- Export data to CSV/JSON for analysis
- Connect to visualization tools
- Integrate with ML experiment tracking platforms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for ChatGPT API
- Anthropic for Claude API  
- Google for Gemini API
- Pandoc for markdown to HTML conversion
