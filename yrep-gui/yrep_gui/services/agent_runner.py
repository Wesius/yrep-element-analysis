"""OpenAI-based agent execution for workflow automation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore[import-unresolved]
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv  # type: ignore[import-unresolved]
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# Load .env file from project root
def _load_env_file() -> None:
    """Load environment variables from .env file if it exists."""
    if not DOTENV_AVAILABLE:
        return

    # Try to find .env file in current directory or parent directories
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return


# Load .env on module import
_load_env_file()


@dataclass
class AgentMessage:
    """Represents a message in the agent conversation."""
    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class AgentResult:
    """Result from an agent execution."""
    output: str
    reasoning: str
    tool_calls: List[Dict[str, Any]]
    raw_response: Any


class AgentExecutionError(RuntimeError):
    """Raised when agent execution fails."""


class OpenAIAgent:
    """Executes LLM-based reasoning workflows using OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-pro",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        if not OPENAI_AVAILABLE:
            raise AgentExecutionError("OpenAI package not installed. Run: pip install openai")

        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        if self.model not in ["gpt-5-pro", "gpt-5"]:
            # can only set temperature for non-gpt-5 models
            self.temperature = temperature
            self.max_tokens = max_tokens


        self.conversation_history: List[AgentMessage] = []

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the agent."""
        self.conversation_history = [AgentMessage(role="system", content=prompt)]

    def execute(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Execute a single agent turn with the given input.

        Args:
            user_input: The user's query or instruction
            context: Optional context data to include in the prompt

        Returns:
            AgentResult with the agent's response
        """
        # Build the prompt with context if provided
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"
            full_input = f"{user_input}{context_str}"
        else:
            full_input = user_input

        # Add user message to history
        self.conversation_history.append(AgentMessage(role="user", content=full_input))

        try:
            # Call OpenAI API
            messages = [{"role": msg.role, "content": msg.content} for msg in self.conversation_history]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **({"temperature": self.temperature, "max_tokens": self.max_tokens} 
                   if self.model not in ["gpt-5-pro", "gpt-5"] else {}),
            )

            # Extract response
            assistant_message = response.choices[0].message
            output = assistant_message.content or ""

            # Add assistant response to history
            self.conversation_history.append(AgentMessage(role="assistant", content=output))

            # Parse tool calls if present (for function calling models)
            tool_calls = []
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "function": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                    for tc in assistant_message.tool_calls
                ]

            return AgentResult(
                output=output,
                reasoning=output,  # For basic models, output IS the reasoning
                tool_calls=tool_calls,
                raw_response=response,
            )

        except Exception as exc:
            raise AgentExecutionError(f"OpenAI API call failed: {exc}") from exc

    def clear_history(self) -> None:
        """Clear conversation history except system prompt."""
        if self.conversation_history and self.conversation_history[0].role == "system":
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []


class WorkflowBuilderAgent:
    """
    Agent that generates spectroscopy workflow graphs based on natural language descriptions.

    This agent understands the available node types and can create complete processing
    pipelines by generating valid graph JSON that can be loaded into the GUI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
    ):
        self.agent = OpenAIAgent(api_key=api_key, model=model, temperature=0.3)
        self._setup_system_prompt()

    def _setup_system_prompt(self) -> None:
        """Configure the agent to build workflows."""
        system_prompt = """You are a spectroscopy workflow builder. Your job is to create
node-based processing graphs for spectral analysis pipelines.

Available node types:
- load_signal: Load a single spectrum file (config: path)
- load_signal_batch: Load multiple spectra from directory (config: directory)
- load_references: Load reference spectra (config: directory, element_only)
- trim: Trim wavelength range (config: min_nm, max_nm)
- mask: Mask specific intervals (config: intervals)
- resample: Resample signal (config: n_points, step_nm)
- subtract_background: Subtract background (config: align, optional background input)
- continuum_remove_arpls: ArPLS continuum removal (config: strength)
- continuum_remove_rolling: Rolling continuum removal (config: strength)
- average_signals: Average multiple signals (config: n_points, accepts multiple inputs)
- build_templates: Build detection templates (config: fwhm_nm, species_filter, bands_kwargs)
- shift_search: Align signal to templates (config: spread_nm, iterations)
- detect_nnls: Detect species using NNLS (config: presence_threshold, min_bands)
- plot_signal: Visualize signal (config: title, normalize)

Output format: Valid JSON matching this structure:
{
  "version": 1,
  "nodes": [
    {"id": 1, "identifier": "load_signal", "config": {"path": "data/signal.txt"}, "position": [100, 100]},
    {"id": 2, "identifier": "trim", "config": {"min_nm": 300, "max_nm": 600}, "position": [300, 100]}
  ],
  "edges": [
    {"source": 1, "source_port": 0, "target": 2, "target_port": 0}
  ]
}

Rules:
1. Node IDs must be unique integers
2. Position nodes in a logical left-to-right flow (x spacing ~200-250, y spacing ~150)
3. Edges connect source output port (usually 0) to target input port (usually 0)
4. Common preprocessing order: load → trim → continuum_remove → resample
5. Detection pipeline: signal + references → build_templates → detect_nnls
6. Use subtract_background when background correction is needed
7. Use shift_search before detect_nnls for better alignment

When user asks for a workflow, respond ONLY with valid JSON, no explanations."""

        self.agent.set_system_prompt(system_prompt)

    def build_workflow(self, description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a workflow graph from natural language description.

        Args:
            description: What the user wants to accomplish
            context: Optional context (file paths, parameters, constraints)

        Returns:
            Graph JSON that can be loaded via NodeEditor.load_graph_data()
        """
        prompt = f"Create a spectroscopy workflow for: {description}"
        if context:
            prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
        prompt += "\n\nReturn only the JSON graph, no other text."

        result = self.agent.execute(prompt)

        # Try to parse JSON from response
        output = result.output.strip()

        # Remove markdown code blocks if present
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            if output.startswith("json"):
                output = output[4:].strip()

        try:
            graph_data = json.loads(output)
            return graph_data
        except json.JSONDecodeError as exc:
            raise AgentExecutionError(f"Agent did not return valid JSON: {exc}\n\nOutput: {output}") from exc
