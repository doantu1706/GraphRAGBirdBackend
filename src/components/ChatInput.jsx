import { useState } from "react";

function ChatInput({ onSend, disabled }) {
    const [input, setInput] = useState("");

    const handleSend = () => {
        // Chỉ gửi nếu có nội dung và không bị khóa
        if (!input.trim() || disabled) return;
        
        onSend(input);
        setInput("");
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter") {
            handleSend();
        }
    };

    return (
        // Đổi className thành 'chat-input-container' để khớp với CSS đẹp
        <div className="chat-input-container">
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Nhập câu hỏi về loài chim..."
                disabled={disabled} // Khóa ô nhập khi đang tải
            />
            <button onClick={handleSend} disabled={disabled || !input.trim()}>
                Gửi
            </button>
        </div>
    );
}

export default ChatInput;