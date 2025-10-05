import React, { useEffect, useState } from 'react';
import GrayscaleFlowBackground from './GrayscaleFlowBackground';

// Theme-aware wrapper for the grayscale flow background
// Automatically adjusts colors based on the current theme
export default function ThemeAwareBackground() {
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    // Check current theme from document
    const checkTheme = () => {
      const root = document.documentElement;
      const currentTheme = root.getAttribute('data-theme') || 'light';
      setTheme(currentTheme);
    };

    // Check theme on mount
    checkTheme();

    // Watch for theme changes
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });

    return () => observer.disconnect();
  }, []);

  // Theme-specific color schemes
  const lightThemeConfig = {
    motion: 0.05,
    scale: 1.3,
    lineGap: 0.2,
    lineThickness: 0.03,
    bg: 0.95,      // Very light background
    ink: 0.88      // Slightly darker lines
  };

  const darkThemeConfig = {
    motion: 0.05,
    scale: 1.3,
    lineGap: 0.2,
    lineThickness: 0.03,
    bg: 0.08,      // Very dark background
    ink: 0.15      // Slightly lighter lines
  };

  const config = theme === 'dark' ? darkThemeConfig : lightThemeConfig;

  return <GrayscaleFlowBackground {...config} />;
}
