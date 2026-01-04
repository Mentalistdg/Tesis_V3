import { NavLink, Outlet } from 'react-router-dom';
import {
  Home,
  GitCompare,
  Search,
  List,
  AlertTriangle,
  Thermometer,
  DollarSign,
  Database,
  Target
} from 'lucide-react';
import { clsx } from 'clsx';
import logoImg from '../assets/logo.png';

const navItems = [
  { path: '/signals', label: 'Signals', icon: Target },
  { path: '/overview', label: 'Overview', icon: Home },
  { path: '/compare', label: 'Compare', icon: GitCompare },
  { path: '/detail', label: 'Detail', icon: Search },
  { path: '/trades', label: 'Trades', icon: List },
  { path: '/risk', label: 'Risk', icon: AlertTriangle },
  { path: '/regime', label: 'Regime', icon: Thermometer },
  { path: '/costs', label: 'Costs', icon: DollarSign },
  { path: '/data', label: 'Data', icon: Database },
];

export default function Layout() {
  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <header className="bg-[#0a0a0a] border-b border-[#222222] px-6 py-4">
        <div className="flex items-center justify-center">
          <img src={logoImg} alt="CRONOS" className="h-14 w-auto" />
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-[#0a0a0a] border-b border-[#222222]">
        <div className="flex overflow-x-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-2 px-5 py-3.5 text-sm font-medium whitespace-nowrap transition-all border-b-2',
                  isActive
                    ? 'text-white border-[#c41e3a] bg-[#111111]'
                    : 'text-[#737373] border-transparent hover:text-[#a3a3a3] hover:bg-[#111111]/50'
                )
              }
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </NavLink>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <main className="p-6">
        <Outlet />
      </main>
    </div>
  );
}
