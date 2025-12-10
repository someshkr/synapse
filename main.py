# from src.simple_agent import get_agent
from src.agent import get_agent

def main():
    print("="*60)
    print("👟 Getting started with Project Synapse 🧬 ")
    print("="*60)

    try:
        agent_executor = get_agent()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print("\n✅ Engaging Neural Engine with Synapse. Ask me to run the forecast.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        print("\n🧠 Processing on Neural Engine...\n")
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()