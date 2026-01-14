import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import ScanDashboard from './components/ScanDashboard';
import ScanHistory from './components/ScanHistory';
import AuditLog from './components/AuditLog';
import { LayoutDashboard, History, FileText } from 'lucide-react';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100 flex">
        {/* Sidebar */}
        <aside className="w-64 bg-slate-900 text-white flex flex-col">
          <div className="p-6">
            <h1 className="text-xl font-bold flex items-center gap-2">
              🛡️ Fortress 95 Pro
            </h1>
            <p className="text-xs text-slate-400 mt-1">v9.4 Migration</p>
          </div>

          <nav className="flex-1 px-4 space-y-2">
            <Link to="/" className="flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-slate-800 transition-colors">
              <LayoutDashboard size={20} />
              <span>Live Scanner</span>
            </Link>
            <Link to="/history" className="flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-slate-800 transition-colors">
              <History size={20} />
              <span>History Intelligence</span>
            </Link>
            <Link to="/audit" className="flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-slate-800 transition-colors">
              <FileText size={20} />
              <span>Audit Logs</span>
            </Link>
          </nav>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<ScanDashboard />} />
            <Route path="/history" element={<ScanHistory />} />
            <Route path="/audit" element={<AuditLog />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
