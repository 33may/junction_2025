import React from 'react';
import './ThemeToggle.css';

const ThemeToggle = ({ theme, onThemeChange }) => {
const themes = [
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
  { value: 'system', label: 'System' }
];

  return (
    <div className="theme-toggle">
      <div className="theme-buttons">
        {themes.map((themeOption) => (
          <button
            key={themeOption.value}
            className={`theme-button ${theme === themeOption.value ? 'active' : ''}`}
            onClick={() => onThemeChange(themeOption.value)}
            title={themeOption.label}
          >
            <span className="theme-label">{themeOption.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default ThemeToggle;
