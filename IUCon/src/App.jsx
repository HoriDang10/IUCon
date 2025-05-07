import { useEffect, useState } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const res = await axios.get("http://localhost:5000/api/users");
        console.log("Users fetched:", res.data.users);
      } catch (err) {
        console.error("Failed to fetch users:", err);
      }
    };
  
    fetchUsers();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const date = new Date();
    const time = `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`;

    // Add user's message
    const userMsg = {
      sender: 'user',
      content: text,
      time: time,
    };
    setMessages(prev => [...prev, userMsg]);
    setText('');

    try {
      const res = await axios.post('http://localhost:5000/predict', {
        message: text
      }, {
        headers: { 'Content-Type': 'application/json' }
      });

      const botMsg = {
        sender: 'bot',
        content: res.data.answer,
        time: time,
      };
      setMessages(prev => [...prev, botMsg]);

    } catch (err) {
      console.error('Error:', err);
    }
  };

  return (
    <div className="container-fluid h-100">
      <div className="row justify-content-center h-100">
        <div className="col-md-8 col-xl-6 chat">
          <div className="card">
            <div className="card-header msg_head">
              <div className="d-flex bd-highlight">
                <div className="img_cont">
                  <img src="https://ibb.co/BYJM89M/logo.png" className="rounded-circle user_img" alt="User avatar" />
                  <span className="online_icon"></span>
                </div>
                <div className="user_info">
                  <span>IU Consultant</span>
                  <p>Hãy hỏi tôi bất cứ điều gì!</p>
                </div>
              </div>
            </div>

            <div className="card-body msg_card_body">
              {messages.map((msg, i) => (
                <div key={i} className={`d-flex mb-4 ${msg.sender === 'user' ? 'justify-content-end' : 'justify-content-start'}`}>
                  {msg.sender === 'bot' && (
                    <div className="img_cont_msg">
                      <img src="https://ibb.co/BYJM89M/logo.png" className="rounded-circle user_img_msg" alt="Bot" />
                    </div> 
                  )}
                  <div className={msg.sender === 'user' ? 'msg_cotainer_send' : 'msg_cotainer'}>
                    {msg.content}
                    <span className="msg_time_send">{msg.time}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="card-footer">
              <form className="input-group" onSubmit={handleSubmit}>
                <input
                  type="text"
                  placeholder="Type your message..."
                  className="form-control type_msg"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  required
                />
                <div className="input-group-append">
                  <button type="submit" className="input-group-text send_btn">
                    <i className="fas fa-location-arrow"></i>
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
