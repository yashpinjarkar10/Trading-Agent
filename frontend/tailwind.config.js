/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: '#070810',
          elevated: '#0d0f1a',
          subtle: '#11141f',
          card: 'rgba(20, 23, 36, 0.55)',
        },
        border: {
          DEFAULT: 'rgba(255, 255, 255, 0.06)',
          strong: 'rgba(255, 255, 255, 0.12)',
        },
        text: {
          primary: '#f5f7fb',
          secondary: '#a3a8bd',
          muted: '#5e6478',
        },
        accent: {
          cyan: '#22d3ee',
          violet: '#8b5cf6',
          green: '#10b981',
          red: '#ef4444',
          amber: '#f59e0b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      backgroundImage: {
        'grid-fade':
          'radial-gradient(ellipse at top, rgba(34,211,238,0.08), transparent 60%), radial-gradient(ellipse at bottom, rgba(139,92,246,0.06), transparent 60%)',
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(34,211,238,0.25), 0 8px 32px -8px rgba(34,211,238,0.35)',
        card: '0 1px 0 0 rgba(255,255,255,0.04) inset, 0 8px 24px -12px rgba(0,0,0,0.6)',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },
        pulse_soft: {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '1' },
        },
      },
      animation: {
        shimmer: 'shimmer 2s linear infinite',
        'pulse-soft': 'pulse_soft 2.5s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
