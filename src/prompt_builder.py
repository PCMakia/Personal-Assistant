build_sec_prompt(user_input: str, recent_messages: list[dict], conversation_summary: str = "", instruction: str = "") {
    fin_prompt = "[system role: A helpful secretary that organize and set schedule for the boss]; [Conversation summary: {conversation_summary}]; [Recent conversation: {recent_conversation}]; [User input: {user_input}]; [Instruction: {instruction}]"
    
    return fin_prompt
}