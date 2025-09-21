import { Route, BrowserRouter as Router, Routes } from "react-router-dom";
import './App.css';
import ObjectDetectionComponent from "./components/ObjectDetectionComponent";

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<ObjectDetectionComponent />} />
            </Routes>
        </Router>
    );
}

export default App;
