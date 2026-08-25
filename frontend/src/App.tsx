import { NavLink, Route, Routes } from "react-router-dom";
import Archive from "./pages/Archive";
import Identify from "./pages/Identify";
import Methodology from "./pages/Methodology";
import Overview from "./pages/Overview";
import Predict from "./pages/Predict";

export default function App() {
  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          <strong>CYCLONESENSE</strong>
          <span>NIO pattern intelligence</span>
        </div>
        <nav>
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/identify">Identify & classify</NavLink>
          <NavLink to="/predict">Track forecast</NavLink>
          <NavLink to="/archive">Storm archive</NavLink>
          <NavLink to="/method">Methodology</NavLink>
        </nav>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/identify" element={<Identify />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/archive" element={<Archive />} />
          <Route path="/method" element={<Methodology />} />
        </Routes>
      </main>
    </div>
  );
}
