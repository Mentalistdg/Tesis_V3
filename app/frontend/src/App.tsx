import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import SignalsPage from './pages/SignalsPage';
import OverviewPage from './pages/OverviewPage';
import ComparePage from './pages/ComparePage';
import DetailPage from './pages/DetailPage';
import TradesPage from './pages/TradesPage';
import RiskPage from './pages/RiskPage';
import RegimePage from './pages/RegimePage';
import CostsPage from './pages/CostsPage';
import DataPage from './pages/DataPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/signals" replace />} />
          <Route path="signals" element={<SignalsPage />} />
          <Route path="overview" element={<OverviewPage />} />
          <Route path="compare" element={<ComparePage />} />
          <Route path="detail/:modelName?" element={<DetailPage />} />
          <Route path="trades/:modelName?" element={<TradesPage />} />
          <Route path="risk/:modelName?" element={<RiskPage />} />
          <Route path="regime" element={<RegimePage />} />
          <Route path="costs" element={<CostsPage />} />
          <Route path="data" element={<DataPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
