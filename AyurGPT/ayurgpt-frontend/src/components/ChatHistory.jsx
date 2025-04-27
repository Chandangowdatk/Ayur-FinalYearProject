import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';

const ChatHistory = ({ onSelectChat }) => {
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { isAuthenticated } = useAuth();

  useEffect(() => {
    const fetchChatHistory = async () => {
      if (!isAuthenticated) return;
      
      setLoading(true);
      try {
        const response = await axios.get('/api/chat-history/');
        setChatHistory(response.data);
      } catch (err) {
        setError('Failed to load chat history. Please try again later.');
        console.error('Error fetching chat history:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchChatHistory();
  }, [isAuthenticated]);

  if (!isAuthenticated) {
    return null;
  }

  if (loading) {
    return <div className="chat-history-loading">Loading chat history...</div>;
  }

  if (error) {
    return <div className="chat-history-error">{error}</div>;
  }

  if (chatHistory.length === 0) {
    return (
      <div className="chat-history-empty">
        <p>No chat history yet. Start a new conversation!</p>
      </div>
    );
  }

  return (
    <div className="chat-history-container">
      <h3>Recent Conversations</h3>
      <ul className="chat-history-list">
        {chatHistory.map((chat) => (
          <li 
            key={chat.id} 
            className="chat-history-item"
            onClick={() => onSelectChat && onSelectChat(chat)}
          >
            <div className="chat-question">{chat.question}</div>
            <div className="chat-timestamp">
              {new Date(chat.timestamp).toLocaleString()}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ChatHistory; 