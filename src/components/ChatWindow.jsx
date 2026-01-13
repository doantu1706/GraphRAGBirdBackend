import { useEffect, useRef } from "react";

const ChatWindow = ({ messages, isLoading }) => {
  const messagesEndRef = useRef(null);

  // Tự động cuộn xuống dòng cuối cùng khi có tin nhắn mới
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Hàm xử lý hiển thị nội dung (Biến text thành Ảnh hoặc Link)
  const renderContent = (text) => {
    // Tách từng dòng để xử lý riêng
    return text.split('\n').map((line, index) => {
      
      // 1. KIỂM TRA ẢNH: Cú pháp ![Alt](URL)
      // Regex này bắt các chuỗi bắt đầu bằng ![...](...)
      const imgRegex = /!\[(.*?)\]\((.*?)\)/;
      const imgMatch = line.match(imgRegex);

      if (imgMatch) {
        // Nếu tìm thấy ảnh, trả về thẻ <img>
        return (
          <div key={index} style={{ textAlign: 'center', margin: '10px 0' }}>
            <img 
              src={imgMatch[2]} 
              alt={imgMatch[1]} 
              className="msg-image" // Class này đã được style trong Chat.css
              onError={(e) => e.target.style.display = 'none'} // Ẩn nếu ảnh lỗi
            />
          </div>
        );
      }

      // 2. KIỂM TRA LINK: Cú pháp [Text](URL)
      // Regex này bắt các chuỗi [...](...) nhưng KHÔNG có dấu chấm than ! ở trước
      const parts = [];
      const linkRegex = /(?<!\!)\[(.*?)\]\((.*?)\)/g;
      let lastIndex = 0;
      let match;

      while ((match = linkRegex.exec(line)) !== null) {
        // Thêm phần text thường trước link
        if (match.index > lastIndex) {
          parts.push(line.substring(lastIndex, match.index));
        }
        // Thêm phần link (biến thành thẻ <a>)
        parts.push(
          <a 
            key={match.index} 
            href={match[2]} 
            target="_blank" 
            rel="noopener noreferrer" 
            className="msg-link" // Class này đã được style trong Chat.css
          >
            {match[1]} 🔗
          </a>
        );
        lastIndex = linkRegex.lastIndex;
      }
      // Thêm phần text còn lại sau link
      if (lastIndex < line.length) {
        parts.push(line.substring(lastIndex));
      }

      // Trả về dòng text đã được xử lý (hoặc text thường nếu không có link)
      return <div key={index} style={{ minHeight: '1.2em' }}>{parts.length ? parts : line}</div>;
    });
  };

  return (
    <div className="chat-window">
      <div className="messages-container">
        {/* Lặp qua danh sách tin nhắn để hiển thị */}
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            {renderContent(m.content)}
          </div>
        ))}

        {/* Bong bóng Loading (Chỉ hiện khi đang chờ API) */}
        {isLoading && (
          <div className="loading-bubble">
            <span style={{ marginRight: '8px' }}>⏳</span> 
            Đang tìm kiếm thông tin...
          </div>
        )}
        
        {/* Điểm neo để tự động cuộn xuống */}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default ChatWindow;