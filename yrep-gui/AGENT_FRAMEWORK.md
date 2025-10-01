# Agentic Framework for YREP Spectral Analysis

## Overview

The YREP GUI now includes an **agentic framework** that integrates OpenAI's language models to provide intelligent workflow automation, analysis, and reporting capabilities. This allows you to build sophisticated spectroscopy pipelines that include AI-powered reasoning and decision-making.

## Architecture

### Components

1. **Agent Runner** (`yrep_gui/services/agent_runner.py`)
   - `OpenAIAgent`: Low-level OpenAI API integration with conversation management
   - `WorkflowAgent`: High-level agent with task-specific system prompts
   - `AgentResult`: Structured output from agent executions

2. **Agent Nodes** (in node registry)
   - **Agent: Analyze Signal** - Quality control and signal analysis
   - **Agent: Analyze Detections** - Detection result interpretation
   - **Agent: Suggest Workflow** - Workflow recommendation engine
   - **Agent: Generate Report** - Professional report generation

3. **GUI Integration** (`ui/inspector.py`)
   - Special UI controls for agent configuration:
     - Model selector dropdown (GPT-4, GPT-3.5, etc.)
     - Task type selector
     - API key input (password-protected)
     - Custom prompt editor

## Agent Node Types

### 1. Agent: Analyze Signal
**Purpose**: Analyze spectral signal characteristics and provide quality control feedback.

**Inputs**: Signal

**Outputs**: Analysis (string)

**Configuration**:
- `api_key`: OpenAI API key (optional, uses `OPENAI_API_KEY` env var if empty)
- `model`: Model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.)
- `task`: Analysis mode (quality_control, general, parameter_optimizer)
- `custom_prompt`: Override default analysis with custom instructions

**Use Cases**:
- Automated signal quality assessment
- Detection of artifacts, noise, or saturation
- Preprocessing recommendations
- Data validation

**Example Output**:
```
Signal Quality Assessment:
1. The signal shows good quality with 1500 data points covering 300-600nm
2. Minor baseline drift detected - recommend continuum removal
3. Suggested preprocessing: Apply arPLS continuum removal (strength=0.5)
```

### 2. Agent: Analyze Detections
**Purpose**: Interpret detection results, identify patterns, and suggest validation steps.

**Inputs**: DetectionResult

**Outputs**: Analysis (string)

**Configuration**:
- `api_key`: OpenAI API key
- `model`: Model to use
- `task`: Analysis mode
- `custom_prompt`: Custom analysis instructions

**Use Cases**:
- Detection confidence interpretation
- False positive identification
- Species co-occurrence analysis
- Result validation recommendations

**Example Output**:
```
Detection Analysis:
1. Found 3 species: Cu (0.92), Fe (0.78), Al (0.45)
2. High confidence for Cu and Fe, moderate for Al
3. Cu-Fe co-occurrence suggests metallic sample
4. Recommend validation: Check Al detection against reference spectra
```

### 3. Agent: Suggest Workflow
**Purpose**: Recommend processing workflows based on signal characteristics and context.

**Inputs**: Signal

**Outputs**: Analysis (string)

**Configuration**:
- `api_key`: OpenAI API key
- `model`: Model to use
- `context`: Additional context as JSON (sample type, goals, constraints)
- `custom_prompt`: Custom workflow request

**Use Cases**:
- Adaptive pipeline generation
- Parameter optimization guidance
- Method selection for specific sample types
- Troubleshooting poor detection results

**Example Output**:
```
Recommended Workflow:
1. Trim to 350-550nm (focus on visible range)
2. Apply rolling continuum removal (strength=0.6)
3. Resample to 2000 points for consistency
4. Build templates with FWHM=0.8nm
5. Run NNLS detection with threshold=0.03

Rationale: Your signal has strong continuum baseline and covers UV-Vis range
suitable for elemental detection. Moderate continuum removal preserves features
while reducing baseline effects.
```

### 4. Agent: Generate Report
**Purpose**: Generate professional scientific reports from pipeline results.

**Inputs**: DetectionResult

**Outputs**: Report (string)

**Configuration**:
- `api_key`: OpenAI API key
- `model`: Model to use
- `task`: report_generator
- `include_signal_stats`: Include signal statistics in report

**Use Cases**:
- Automated lab report generation
- Results documentation
- Summary reports for batches
- Quality assurance documentation

**Example Output**:
```markdown
# Spectroscopy Analysis Report

## Executive Summary
Elemental detection analysis identified 3 species with high confidence,
indicating a copper-iron metallic alloy sample.

## Detected Species
- **Copper (Cu)**: 0.92 confidence, 12 bands matched
- **Iron (Fe)**: 0.78 confidence, 8 bands matched
- **Aluminum (Al)**: 0.45 confidence, 5 bands matched

## Signal Quality
- Wavelength range: 300-600nm
- 1500 spectral points
- Signal-to-noise ratio: Good

## Conclusions
The detection results strongly indicate a copper-iron alloy composition.
Aluminum presence is uncertain and requires further validation.

## Recommendations
1. Verify Al detection with independent method
2. Consider quantitative analysis for Cu/Fe ratio
3. Check reference library completeness
```

## Usage Examples

### Example 1: Quality Control Pipeline
```
[Load Signal] → [Agent: Analyze Signal] → [Log Output]
```
Automatically analyze incoming signals for quality issues before processing.

### Example 2: Adaptive Processing
```
[Load Signal] → [Agent: Suggest Workflow] → [Human Review] → [Process Signal]
```
Get AI recommendations for how to process unfamiliar sample types.

### Example 3: Automated Reporting
```
[Load Signal] → [Preprocess] → [Build Templates] → [Detect] → [Agent: Generate Report] → [Save Report]
```
Complete analysis with automatic professional report generation.

### Example 4: Multi-Stage Analysis
```
[Load Signal]
  → [Agent: Analyze Signal] → [Decide Preprocessing]
  → [Preprocess] → [Detect]
  → [Agent: Analyze Detections] → [Validation Steps]
  → [Agent: Generate Report]
```
Comprehensive workflow with AI guidance at multiple decision points.

## Configuration

### API Key Setup

**Option 1: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="sk-..."
```
Leave the `api_key` field empty in node configuration.

**Option 2: Per-Node Configuration**
Enter your API key directly in the Inspector panel. Keys are stored in the graph file (use caution with version control).

### Model Selection

Available models:
- **gpt-4**: Most capable, slower, higher cost
- **gpt-4-turbo**: Fast GPT-4 variant
- **gpt-4o**: Optimized for speed and cost
- **gpt-4o-mini**: Fastest, most economical
- **gpt-3.5-turbo**: Fast and economical, less capable

Choose based on your requirements:
- Complex analysis → GPT-4
- Rapid prototyping → GPT-4o-mini
- Production workflows → GPT-4o

### Task Types

Predefined system prompts for common use cases:
- **general**: Versatile analysis assistant
- **quality_control**: Focus on signal quality and validation
- **report_generator**: Professional scientific writing
- **parameter_optimizer**: Parameter tuning recommendations

### Custom Prompts

Override default behavior with custom instructions:
```
Analyze this signal and determine if it's suitable for mineral detection.
Focus on spectral resolution and noise characteristics.
Provide specific preprocessing recommendations.
```

## Installation

Install the OpenAI Python package:
```bash
pip install openai
```

Or add to your `pyproject.toml`:
```toml
[project]
dependencies = [
    "openai>=1.0.0",
]
```

## Best Practices

1. **Start Simple**: Begin with default configurations and tasks
2. **Iterative Refinement**: Use custom prompts to tune agent behavior
3. **Context is Key**: Provide rich context in the `context` field for better suggestions
4. **Validate AI Outputs**: Always verify agent recommendations before applying
5. **Cost Management**: Use cheaper models for prototyping, upgrade for production
6. **Error Handling**: Agent nodes will fail gracefully with informative errors
7. **Privacy**: Be mindful of sending sensitive data to OpenAI's API

## Troubleshooting

### "OpenAI package not installed"
Run: `pip install openai`

### "Agent execution failed: Authentication error"
- Check your API key is correct
- Verify `OPENAI_API_KEY` environment variable
- Ensure API key has sufficient credits

### "Rate limit exceeded"
- Slow down request rate
- Upgrade your OpenAI plan
- Add retry logic in workflows

### Agent outputs seem generic
- Add more context in configuration
- Use custom prompts to specify exactly what you want
- Try GPT-4 instead of GPT-3.5

## Future Enhancements

Potential extensions to the framework:
- **Multi-agent workflows**: Agents that delegate to specialized sub-agents
- **Tool calling**: Agents that can invoke pipeline functions directly
- **Feedback loops**: Agents that iteratively refine parameters
- **Local LLM support**: Integration with Ollama, LLaMA, etc.
- **Agent memory**: Persistent context across runs
- **Batch processing**: Parallel agent execution for large datasets

## API Reference

### WorkflowAgent

```python
class WorkflowAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        task: str = "general"
    )

    def analyze_signal(self, signal_data: Dict[str, Any]) -> str
    def analyze_detections(self, detections: List[Dict[str, Any]]) -> str
    def suggest_workflow(self, context: Dict[str, Any]) -> str
    def generate_report(self, pipeline_results: Dict[str, Any]) -> str
```

### OpenAIAgent

```python
class OpenAIAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000
    )

    def execute(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult

    def set_system_prompt(self, prompt: str) -> None
    def clear_history(self) -> None
```

---

**Generated for YREP Spectral Node Editor**
Version 1.0 | OpenAI Integration
