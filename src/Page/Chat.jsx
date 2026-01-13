import { useState } from "react";
import Sidebar from "../components/Sidebar";
import ChatWindow from "../components/ChatWindow";
import ChatInput from "../components/ChatInput";
import "./Chat.css";

function Chat() {
    const [conversations, setConversations] = useState([
        {
            id: "1",
            title: "Chào mừng",
            messages: [
                { role: "assistant", content: "Xin chào! Bạn muốn đặt câu hỏi gì về các loài chim?" }
            ]
        }
    ]);

    const [activeId, setActiveId] = useState("1");
    const [isLoading, setIsLoading] = useState(false);
    const activeConv = conversations.find(c => c.id === activeId);

    const sendMessage = async (text) => {
        if (!text.trim()) return;

        const userMsg = { role: "user", content: text };

        // Cập nhật tin nhắn user lên màn hình trước
        setConversations(prev =>
            prev.map(conv =>
                conv.id === activeId
                    ? { ...conv, messages: [...conv.messages, userMsg] }
                    : conv
            )
        );

        setIsLoading(true);

        try {
            // --- QUAN TRỌNG: SỬA LINK VỀ LOCALHOST ---
            console.log("Đang gửi tin nhắn đến Backend..."); // Log để kiểm tra
            
            const response = await fetch("http://localhost:5000/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ 
                    message: text,
                    session_id: activeId 
                }),
            });

            // Kiểm tra nếu server báo lỗi (ví dụ 404, 500)
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            
            const botMsg = { 
                role: "assistant", 
                content: data.response || "⚠️ Server không trả về nội dung." 
            };

            setConversations(prev =>
                prev.map(conv =>
                    conv.id === activeId
                        ? { ...conv, messages: [...conv.messages, botMsg] }
                        : conv
                )
            );

        } catch (error) {
            console.error("Lỗi kết nối chi tiết:", error);
            const errorMsg = { 
                role: "assistant", 
                content: "⚠️ Lỗi: Không kết nối được Backend. Hãy kiểm tra xem cửa sổ Python có đang chạy không?" 
            };
            
            setConversations(prev =>
                prev.map(conv =>
                    conv.id === activeId
                        ? { ...conv, messages: [...conv.messages, errorMsg] }
                        : conv
                )
            );
        } finally {
            setIsLoading(false);
        }
    };

    const newChat = () => {
        const newId = Date.now().toString();
        const newConversation = {
            id: newId,
            title: "Cuộc trò chuyện mới",
            messages: []
        };
        setConversations(prev => [newConversation, ...prev]);
        setActiveId(newId);
    };

    return (
        <div className="app">
            <div className="sidebar-container">
                <Sidebar
                    conversations={conversations}
                    activeId={activeId}
                    onSelect={setActiveId}
                    onNewChat={newChat}
                />
            </div>
            
            <div className="chat-layout-main">
                {/* 1. HEADER MỚI Ở ĐÂY */}
                <div className="chat-header">
                    Hệ thống tra cứu chim 🦜
                </div>

                <div className="chat-main">
                    {/* 2. CHAT WINDOW (Hiển thị tin nhắn + Loading) */}
                    <ChatWindow 
                        messages={activeConv?.messages || []} 
                        isLoading={isLoading} 
                    />
                    
                    {/* 3. INPUT (Được bọc div wrapper để căn giữa) */}
                    <div className="chat-input-wrapper">
                        <ChatInput onSend={sendMessage} disabled={isLoading} />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Chat;