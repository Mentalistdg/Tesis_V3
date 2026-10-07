import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import OverviewPage from './pages/OverviewPage';
import DetailPage from './pages/DetailPage';
import TradesPage from './pages/TradesPage';
import RiskPage from './pages/RiskPage';
import RegimePage from './pages/RegimePage';
import CostsPage from './pages/CostsPage';
import SenalesPage from './pages/SenalesPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/senales" replace />} />
          <Route path="senales" element={<SenalesPage />} />
          <Route path="overview" element={<OverviewPage />} />
          <Route path="detail/:modelName?" element={<DetailPage />} />
          <Route path="trades/:modelName?" element={<TradesPage />} />
          <Route path="risk/:modelName?" element={<RiskPage />} />
          <Route path="regime" element={<RegimePage />} />
          <Route path="costs" element={<CostsPage />} />
          <Route path="*" element={<Navigate to="/senales" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
