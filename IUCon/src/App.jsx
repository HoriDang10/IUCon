import { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [typingText, setTypingText] = useState('');
  const messageEndRef = useRef(null);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping, typingText]);

  const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

  const handleSubmit = async (e) => {
    e.preventDefault();
    const input = e.target.elements.msg;
    const messageText = input.value;
    input.value = '';

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    setMessages((prev) => [...prev, { text: messageText, sender: 'user', time }]);
    setIsTyping(true);
    setTypingText('');

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText }),
      });

      const data = await response.json();

      let currentText = '';
      const chars = data.answer.split('');

      for (let ch of chars) {
        currentText += ch;
        setTypingText(currentText);
        await sleep(10); // Typing speed per character
      }

      setMessages((prev) => [...prev, { text: currentText, sender: 'bot', time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, { text: "Lỗi hệ thống. Vui lòng thử lại sau.", sender: 'bot', time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
    } finally {
      setIsTyping(false);
      setTypingText('');
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
                  <img src="https://i.ibb.co/mCsb0sHn/logo.png" className="rounded-circle user_img" alt="logo" />
                  <span className="online_icon"></span>
                </div>
                <div className="user_info">
                  <span>IU Consultant</span>
                  <p>Hãy hỏi tôi bất cứ điều gì!</p>
                </div>
              </div>
            </div>
            <div className="card-body msg_card_body" id="messageContainer">
              {messages.map((msg, index) => (
                <div key={index} className={`d-flex mb-4 ${msg.sender === 'user' ? 'justify-content-end' : 'justify-content-start'}`}>
                  {msg.sender === 'bot' && (
                    <div className="img_cont_msg">
                      <img src="https://i.ibb.co/mCsb0sHn/logo.png" className="rounded-circle user_img_msg" alt="bot avatar" />
                    </div>
                  )}
                  <div className={msg.sender === 'user' ? 'msg_cotainer_send' : 'msg_cotainer'} style={{ whiteSpace: 'pre-wrap' }}>
                    <div dangerouslySetInnerHTML={{ __html: msg.text.replace(/\n/g, '<br>') }} />
                    <span className={msg.sender === 'user' ? 'msg_time_send' : 'msg_time'}>{msg.time}</span>
                  </div>
                  {msg.sender === 'user' && (
                    <div className="img_cont_msg">
                      <img src="https://i.ibb.co/V0112Lck/audience.png" className="rounded-circle user_img_msg" alt="user avatar" />
                    </div>
                  )}
                </div>
              ))}

              {isTyping && (
                <div className="d-flex mb-4 justify-content-start">
                  <div className="img_cont_msg">
                    <img src="https://i.ibb.co/mCsb0sHn/logo.png" className="rounded-circle user_img_msg" alt="typing bot" />
                  </div>
                  <div className="msg_cotainer typing" style={{ whiteSpace: 'pre-wrap' }}>{typingText}</div>
                </div>
              )}

              <div ref={messageEndRef}></div>
            </div>
            <div className="card-footer">
              <form className="input-group" onSubmit={handleSubmit}>
                <input
                  type="text"
                  name="msg"
                  placeholder="Type your message..."
                  className="form-control type_msg"
                  autoComplete="off"
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