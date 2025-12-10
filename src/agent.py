import os
import json
import re
import time
from typing import Any, Dict, List

# ✅ Standard Modern Imports
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_agent
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from src.pipeline import SynapsePipeline

# Initialize the pipeline
pipeline_logic = SynapsePipeline()

# ==============================================================================
# 1. DEFINE TOOLS (The Actions)
# ==============================================================================

def run_full_pipeline_tool(query: str) -> str:
    """Run the complete pipeline."""
    print("⚙️ Running complete pipeline...")
    res1 = pipeline_logic.load_and_prep()
    res2 = pipeline_logic.train_and_compare()
    res3 = pipeline_logic.run_causal_analysis()
    res4 = pipeline_logic.generate_forecast()
    return f"Complete Pipeline Results:\n\n{res1}\n\n{res2}\n\n{res3}\n\n{res4}"

def compare_models_tool(query: str) -> str:
    """Compare model performance."""
    print("⚙️ Comparing models...")
    result = pipeline_logic.train_and_compare()
    return f"Model Comparison Results:\n\n{result}"

def causal_analysis_tool(query: str) -> str:
    """Run causal analysis."""
    print("⚙️ Running causal analysis...")
    result = pipeline_logic.run_causal_analysis()
    return f"Causal Analysis Results:\n\n{result}"

def forecast_tool(query: str) -> str:
    """Generate forecast."""
    print("⚙️ Generating forecast...")
    result = pipeline_logic.generate_forecast()
    return f"Forecast Results:\n\n{result}"

# Create tool objects
tools = [
    Tool(
        name="run_full_pipeline",
        func=run_full_pipeline_tool,
        description=(
            "Run the complete pipeline on data.xlsx: load and preprocess the data, "
            "compare models, run causal analysis, and then generate the forecast. "
            "Use when the user wants EVERYTHING end-to-end or asks for 'full analysis'."
        ),
    ),
    Tool(
        name="compare_models",
        func=compare_models_tool,
        description=(
            "Compare model performance metrics (MAPE, MAE, RMSE) for Orbit ETS, Prophet, "
            "and GAM on the Nike sales data. Use when the user asks about model performance, "
            "which model is best, or wants to see comparison metrics."
        ),
    ),
    Tool(
        name="run_causal_analysis",
        func=causal_analysis_tool,
        description=(
            "Run the causal analysis and elasticity diagnostics on the Nike sales data: "
            "GAM elasticities, Double ML rubber price impact, and consumer expenditure impact. "
            "Use when the user asks about drivers, elasticities, causal effects, or what drives sales."
        ),
    ),
    Tool(
        name="generate_forecast",
        func=forecast_tool,
        description=(
            "Generate the final Nike sales forecast for 2026–2030 from data.xlsx. "
            "This will automatically load/prep the data, select the best model by MAPE, "
            "and return: the forecast table, growth from the last observed year, "
            "key driver elasticities, and model performance metrics. "
            "Use whenever the user asks to 'run the forecast', 'generate forecast', "
            "or 'given data.xlsx generate forecast'."
        ),
    )
]

# ==============================================================================
# 2. CUSTOM EXECUTOR (The Brain Fix)
# ==============================================================================

class SimpleAgentExecutor:
    """
    A custom executor that fixes Llama 3's tendency to output JSON as text 
    instead of running the tool. It intercepts the JSON, runs the tool manually, 
    and formats the output.
    """
    def __init__(self, graph, tools):
        self.graph = graph
        self.tools = tools # Access to tools is required for manual execution
        self.messages = []
        
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_input = inputs.get("input", "")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("Agent input must be a non-empty string.")
        
        print(f"🔍 Processing: {user_input}")
        
        # Add user message to history
        self.messages.append(HumanMessage(content=user_input))
        
        try:
            # 1. Ask the LLM what to do
            # We use a short sleep to simulate "thinking" if needed, or remove for speed
            # time.sleep(1) 
            result = self.graph.invoke({"input": user_input, "chat_history": self.messages})
            
            # The 'create_react_agent' returns the final output string directly in 'output'
            # But sometimes Llama 3 puts the JSON in the 'output' string.
            raw_output = result.get("output", "")
            
            # 2. CHECK: Did the model just give us JSON text instead of running the tool?
            # This logic catches the {"name": "generate_forecast"} text.
            json_match = re.search(r'\{.*"name":\s*"(.*?)".*\}', raw_output, re.DOTALL)
            
            if json_match:
                # We found a tool call hidden in the text!
                try:
                    # Extract the JSON blob
                    start = raw_output.find('{')
                    end = raw_output.rfind('}') + 1
                    json_str = raw_output[start:end]
                    
                    tool_data = json.loads(json_str)
                    tool_name = tool_data.get("name")
                    
                    print(f"🔧 Detected JSON intent: '{tool_name}'")
                    
                    # 3. EXECUTE: Find the tool and run it manually
                    target_tool = next((t for t in self.tools if t.name == tool_name), None)
                    
                    if target_tool:
                        print(f"⚙️  Executing tool: {tool_name} (Please wait for training)...")
                        
                        # Run the python function!
                        # We ignore parameters from LLM and pass user input (our tools are robust)
                        tool_output = target_tool.func(user_input)
                        
                        print(f"✅ Execution finished.")
                        return {"output": self._format_tool_response(tool_output)}
                    else:
                        print(f"❌ Tool '{tool_name}' not found.")
                        return {"output": f"Error: Tool {tool_name} not found."}
                        
                except Exception as e:
                    print(f"⚠️ JSON Parse Error: {e}")
                    # If parsing fails, just return the raw text
                    return {"output": raw_output}

            # 4. Standard Case: If no JSON detected, return the text as is
            # (Usually means the model answered a general question)
            return {"output": raw_output}
            
        except Exception as e:
            print(f"❌ Error during agent execution: {e}")
            return {"output": f"An error occurred: {str(e)}"}
    
    def _format_tool_response(self, tool_result: str) -> str:
        """Format the raw tool result into a nice natural language response."""
        
        if "Forecast Results:" in tool_result:
            return f"""Here are the Nike sales forecast results based on your data:

{tool_result}

**Summary:**
I have generated the forecast for 2026-2030. The best model was selected based on historical accuracy (MAPE).
Key drivers (Elasticity) and Confidence Intervals are included above."""

        elif "Model Comparison Results:" in tool_result:
            return f"""Here are the model performance metrics:

{tool_result}

**Interpretation:**
I compared Orbit, Prophet, and GAM. The model with the lowest MAPE (Mean Absolute Percentage Error) is typically the most accurate for this dataset."""

        elif "Causal Analysis Results:" in tool_result:
            return f"""Here is the Causal Analysis:

{tool_result}

**Key Insights:**
These results show the statistical impact of variables like Rubber Price and Consumer Expenditure on Nike's sales figures."""

        elif "Complete Pipeline Results:" in tool_result:
            return f"""Complete Analysis Pipeline Finished:

{tool_result}

**Done:**
All steps (Data Load -> Model Train -> Causal Inference -> Forecasting) have been executed successfully."""

        else:
            return f"Analysis Complete:\n\n{tool_result}"


# ==============================================================================
# 3. AGENT FACTORY
# ==============================================================================

def get_agent():
    print("🍎 Loading Model into Mac M4 GPU/Unified Memory...")
    
    # Update path to your downloaded model
    model_path = "models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Please download the GGUF model and place it at: {model_path}")

    llm = ChatLlamaCpp(
        model_path=model_path,
        temperature=0,
        n_gpu_layers=-1,
        n_ctx=4096,
        max_tokens=4096,
        n_batch=512,
        f16_kv=True,
        verbose=True,
        stop=["Observation:", "Observation"]
    )

    # Create the ReAct Agent Graph
    agent_graph = create_react_agent(llm, tools)
    
    # Updated SimpleAgentExecutor to handle AIMessage objects
    class SimpleAgentExecutor:
        def __init__(self, graph, tools):
            self.graph = graph
            self.tools = tools
            self.messages = []
            
        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            user_input = inputs.get("input", "")
            if not isinstance(user_input, str) or not user_input.strip():
                raise ValueError("Agent input must be a non-empty string.")
            
            print(f"🔍 Processing: {user_input}")
            
            # Add user message to history
            self.messages.append(HumanMessage(content=user_input))
            
            try:
                # 1. Ask the LLM what to do
                # The graph returns an AIMessage object
                result = self.graph.invoke({"messages": self.messages})
                
                # Extract the AIMessage from result
                if isinstance(result, dict) and "messages" in result:
                    # Result is a dict with messages list
                    messages = result["messages"]
                    last_message = messages[-1] if messages else None
                elif isinstance(result, AIMessage):
                    # Result is directly an AIMessage
                    last_message = result
                else:
                    # Try to get the output directly
                    last_message = result
                
                # Get the content from the message
                if hasattr(last_message, 'content'):
                    raw_output = last_message.content
                elif isinstance(last_message, str):
                    raw_output = last_message
                elif isinstance(last_message, dict):
                    raw_output = last_message.get("output", "")
                else:
                    raw_output = str(last_message)
                
                print(f"📄 Agent response: {raw_output[:200]}...")
                
                # 2. CHECK: Did the model just give us JSON text instead of running the tool?
                json_match = re.search(r'\{.*"name":\s*"(.*?)".*\}', raw_output, re.DOTALL)
                
                if json_match:
                    # We found a tool call hidden in the text!
                    try:
                        # Extract the JSON blob
                        start = raw_output.find('{')
                        end = raw_output.rfind('}') + 1
                        json_str = raw_output[start:end]
                        
                        tool_data = json.loads(json_str)
                        tool_name = tool_data.get("name")
                        
                        print(f"🔧 Detected JSON intent: '{tool_name}'")
                        
                        # 3. EXECUTE: Find the tool and run it manually
                        target_tool = next((t for t in self.tools if t.name == tool_name), None)
                        
                        if target_tool:
                            print(f"⚙️  Executing tool: {tool_name} (Please wait for training)...")
                            
                            # Run the python function!
                            tool_output = target_tool.func(user_input)
                            
                            print(f"✅ Execution finished.")
                            return {"output": self._format_tool_response(tool_output)}
                        else:
                            print(f"❌ Tool '{tool_name}' not found.")
                            return {"output": f"Error: Tool {tool_name} not found."}
                            
                    except Exception as e:
                        print(f"⚠️ JSON Parse Error: {e}")
                        # If parsing fails, just return the raw text
                        return {"output": raw_output}

                # 4. Standard Case: If no JSON detected, return the text as is
                return {"output": raw_output}
                
            except Exception as e:
                print(f"❌ Error during agent execution: {e}")
                import traceback
                traceback.print_exc()
                return {"output": f"An error occurred: {str(e)}"}
        
        def _format_tool_response(self, tool_result: str) -> str:
            """Format the raw tool result into a nice natural language response."""
            
            if "Forecast Results:" in tool_result:
                return f"""Here are the Nike sales forecast results based on your data:

{tool_result}

**Summary:**
I have generated the forecast for 2026-2030. The best model was selected based on historical accuracy (MAPE).
Key drivers (Elasticity) and Confidence Intervals are included above."""

            elif "Model Comparison Results:" in tool_result:
                return f"""Here are the model performance metrics:

{tool_result}

**Interpretation:**
I compared Orbit, Prophet, and GAM. The model with the lowest MAPE (Mean Absolute Percentage Error) is typically the most accurate for this dataset."""

            elif "Causal Analysis Results:" in tool_result:
                return f"""Here is the Causal Analysis:

{tool_result}

**Key Insights:**
These results show the statistical impact of variables like Rubber Price and Consumer Expenditure on Nike's sales figures."""

            elif "Complete Pipeline Results:" in tool_result:
                return f"""Complete Analysis Pipeline Finished:

{tool_result}

**Done:**
All steps (Data Load -> Model Train -> Causal Inference -> Forecasting) have been executed successfully."""

            else:
                return f"Analysis Complete:\n\n{tool_result}"
    
    return SimpleAgentExecutor(agent_graph, tools)