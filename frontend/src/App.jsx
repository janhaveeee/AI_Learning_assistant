import { Routes, Route, NavLink } from "react-router-dom";
import Home from "./pages/Home";
import Predict from "./pages/Predict";
import Tutor from "./pages/Tutor";
import DifficultyPredictor from "./pages/DifficultyPredictor";
import StressPrediction from "./pages/StressPrediction";




import './index.css';

function App() {
  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-logo">SkillMentor 🚀</div>
        <div className="nav-links">
          <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active-link" : "nav-link"}>
            Home
          </NavLink>
          <NavLink to="/predict" className={({ isActive }) => isActive ? "nav-link active-link" : "nav-link"}>
            Predict
          </NavLink>
          <NavLink to="/tutor" className={({ isActive }) => isActive ? "nav-link active-link" : "nav-link"}>
            Tutor
          </NavLink>
          <NavLink to="/difficulty" className={({ isActive }) => isActive ? "nav-link active-link" : "nav-link"}>
          DifficultyPredictor
          </NavLink>
          <NavLink to="/stress" className={({ isActive }) => isActive ? "nav-link active-link" : "nav-link"}>
          StressPrediction
          </NavLink>
        </div>
      </nav>

      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/tutor" element={<Tutor />} />
          <Route path="/difficulty" element={<DifficultyPredictor />} />
          <Route path="/stress" element={<StressPrediction />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>© 2025 SkillMentor. All rights reserved.</p>
        <div className="footer-links">
          <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
          <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">LinkedIn</a>
          <a href="mailto:contact@skillmentor.com">Contact</a>
        </div>
      </footer>
    </div>
  );
}

export default App;
