import { useState } from 'react';
import Sidebar from './Sidebar';
import Topbar from './Topbar';

export default function AppShell({ current, onChange, symbol, onSymbolChange, children }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        current={current}
        onChange={onChange}
        collapsed={collapsed}
        onToggle={() => setCollapsed((v) => !v)}
      />
      <div className="flex-1 flex flex-col min-w-0">
        <Topbar symbol={symbol} onSymbolChange={onSymbolChange} />
        <main className="flex-1 overflow-auto">
          <div className="grid-bg absolute inset-0 pointer-events-none -z-10" />
          {children}
        </main>
      </div>
    </div>
  );
}
