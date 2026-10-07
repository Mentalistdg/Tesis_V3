/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Axe Capital Theme - Institutional Black & Red
        'axe-black': '#000000',
        'axe-dark': '#0a0a0a',
        'axe-card': '#111111',
        'axe-card-hover': '#1a1a1a',
        'axe-border': '#222222',
        'axe-border-light': '#333333',

        // Axe Capital Red
        'axe-red': '#c41e3a',
        'axe-red-light': '#dc3545',
        'axe-red-dark': '#9a1830',

        // Steel Grays
        'axe-gray-100': '#f5f5f5',
        'axe-gray-200': '#e5e5e5',
        'axe-gray-300': '#a3a3a3',
        'axe-gray-400': '#737373',
        'axe-gray-500': '#525252',
        'axe-gray-600': '#404040',

        // Accent Colors (for charts)
        'axe-green': '#00c853',
        'axe-green-dark': '#00a844',

        // Legacy mappings for compatibility
        'tv-bg-primary': '#000000',
        'tv-bg-secondary': '#111111',
        'tv-bg-tertiary': '#1a1a1a',
        'tv-text-primary': '#f5f5f5',
        'tv-text-secondary': '#737373',
        'tv-accent-green': '#00c853',
        'tv-accent-red': '#c41e3a',
        'tv-accent-blue': '#c41e3a',
        'tv-border': '#222222',
      },
      fontFamily: {
        'display': ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
