function Sidebar({ conversations, activeId, onSelect, onNewChat }) {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <button className="new-chat" onClick={onNewChat}>
          + Cuộc trò chuyện mới
        </button>
      </div>

      <div className="sidebar-list">
        {conversations.map(conv => (
          <div
            key={conv.id}
            className={`sidebar-item ${conv.id === activeId ? "active" : ""
              }`}
            onClick={() => onSelect(conv.id)}
            title={conv.title}
          >
            {conv.title}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Sidebar;
