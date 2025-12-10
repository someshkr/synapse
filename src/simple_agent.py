import os
from typing import Any, Dict

from src.pipeline import SynapsePipeline

class SynapseAgent:
    """Simple agent that directly routes queries to the appropriate pipeline methods."""
    
    def __init__(self):
        print("🚀 Initializing SYNAPSE Agent...")
        self.pipeline = SynapsePipeline()
        print("✅ Pipeline loaded successfully")
    
    def _route_query(self, query: str) -> str:
        """Route the query to the appropriate pipeline method."""
        query_lower = query.lower()
        
        # Define routing rules
        if any(word in query_lower for word in ["forecast", "predict", "generate", "future", "2026", "2030"]):
            return "forecast"
        elif any(word in query_lower for word in ["compare", "model", "performance", "metric", "mape", "mae", "rmse"]):
            return "compare"
        elif any(word in query_lower for word in ["causal", "elasticity", "driver", "effect", "impact", "what drives"]):
            return "causal"
        elif any(word in query_lower for word in ["everything", "full", "complete", "end-to-end", "all", "pipeline"]):
            return "full"
        else:
            # Default to forecast
            return "forecast"
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input query and return the response."""
        try:
            query = inputs.get("input", "").strip()
            if not query:
                return {"output": "Please provide a question or request."}
            
            print(f"🤖 Processing query: {query}")
            
            # Route the query
            route = self._route_query(query)
            print(f"🛣️  Routing to: {route}")
            
            # Execute appropriate pipeline method
            if route == "forecast":
                print("🔮 Generating forecast...")
                result = self.pipeline.generate_forecast()
            elif route == "compare":
                print("📊 Comparing models...")
                result = self.pipeline.train_and_compare()
            elif route == "causal":
                print("🔍 Running causal analysis...")
                result = self.pipeline.run_causal_analysis()
            elif route == "full":
                print("⚡ Running full pipeline...")
                # Run all components
                result_parts = []
                
                print("📥 Loading data...")
                res1 = self.pipeline.load_and_prep()
                result_parts.append(f"📊 DATA LOADING:\n{res1}")
                
                print("🤖 Comparing models...")
                res2 = self.pipeline.train_and_compare()
                result_parts.append(f"📈 MODEL COMPARISON:\n{res2}")
                
                print("🔬 Running causal analysis...")
                res3 = self.pipeline.run_causal_analysis()
                result_parts.append(f"🔍 CAUSAL ANALYSIS:\n{res3}")
                
                print("🔮 Generating forecast...")
                res4 = self.pipeline.generate_forecast()
                result_parts.append(f"🔮 FORECAST:\n{res4}")
                
                result = "\n\n" + "="*50 + "\n\n".join(result_parts) + "\n" + "="*50
            else:
                result = "I'm not sure how to handle that request. Please ask about forecasts, model comparison, causal analysis, or running the full pipeline."
            
            print("✅ Response generated successfully")
            return {"output": result}
            
        except Exception as e:
            error_msg = f"❌ Error processing your request: {str(e)}"
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {"output": error_msg}

def get_agent():
    """Return the agent instance."""
    return SynapseAgent()