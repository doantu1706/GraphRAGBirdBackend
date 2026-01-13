import { BrowserRouter, Routes, Route } from "react-router-dom";
import WelcomePage from "./Page/Home";
import ChatPage from "./Page/Chat";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/chat" element={<ChatPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
