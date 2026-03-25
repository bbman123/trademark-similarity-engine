import React from 'react';
import { ShieldCheck, BookOpen } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50 backdrop-blur-sm bg-white/95">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-600 to-primary-800 rounded-lg flex items-center justify-center shadow-lg">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Trademark Registration</h1>
              <p className="text-xs text-gray-500">Decision Support System</p>
            </div>
          </div>
          <nav className="hidden md:flex items-center space-x-6">
            <a
              href="https://buhari123-trademark-similarity-api.hf.space/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 text-gray-600 hover:text-primary-600 transition-colors"
            >
              <BookOpen className="w-4 h-4" />
              <span className="text-sm font-medium">API Docs</span>
            </a>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
